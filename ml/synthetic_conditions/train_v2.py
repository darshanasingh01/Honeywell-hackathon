"""Train the separate synthetic-only 167-feature V2 baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .schema_v2 import FEATURE_COLUMNS, MANIFEST, SCHEMA_VERSION, validate_frame
from .verify_producer_v2 import verify_directory

N_SPLITS = 5
SEED = 42
MODEL_VERSION = "synthetic-conditions-167-v2-baseline-v1"
DEFAULT_DATASET = Path("ml/datasets/synthetic_conditions_v2/training")
DEFAULT_REPORT = Path("ml/evaluation/synthetic_conditions_167_v2_groupkfold.json")
DEFAULT_MODEL = Path("ml/saved_models/synthetic_conditions_167_v2_baseline.joblib")
CLASS_LABELS = tuple(MANIFEST["class_labels"])


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_models() -> dict[str, Any]:
    return {
        "random_forest": RandomForestClassifier(n_estimators=300, max_features="sqrt",
            class_weight="balanced", random_state=SEED, n_jobs=1),
        "scaled_rbf_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                                class_weight="balanced")),
        ]),
    }


def _summary(y_true, y_pred) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS,
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
        "class_order": list(CLASS_LABELS),
        "per_class": {label: {"precision": float(precision[i]),
            "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(CLASS_LABELS)},
        "per_class_recall": {label: float(recall[i]) for i, label in enumerate(CLASS_LABELS)},
    }


def grouped_splits(X, y, groups):
    if groups.nunique() < N_SPLITS:
        raise ValueError(f"GroupKFold requires at least {N_SPLITS} unique runs")
    for fold, (train_idx, test_idx) in enumerate(
            GroupKFold(n_splits=N_SPLITS).split(X, y, groups), 1):
        train_ids, test_ids = set(groups.iloc[train_idx]), set(groups.iloc[test_idx])
        if train_ids & test_ids:
            raise RuntimeError(f"run_id leakage in fold {fold}")
        if set(y.iloc[train_idx]) != set(CLASS_LABELS) or set(y.iloc[test_idx]) != set(CLASS_LABELS):
            raise ValueError(f"fold {fold} lacks one or more classes")
        yield fold, train_idx, test_idx


def load_training_handoff(dataset: str | Path = DEFAULT_DATASET):
    dataset = Path(dataset)
    if not dataset.is_dir():
        raise FileNotFoundError(f"V2 dataset directory not found: {dataset}")
    manifest_path, features_path = dataset / "manifest.json", dataset / "features.csv"
    validation_path = dataset / "validation.json"
    if not manifest_path.exists() or not features_path.exists() or not validation_path.exists():
        raise FileNotFoundError("V2 handoff requires features.csv, manifest.json, and validation.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    expected_sha = _sha256(features_path)
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("data_kind") != "synthetic_simulated_engine_conditions"
            or manifest.get("real_engine_measurements") is not False
            or manifest.get("features_sha256") != expected_sha
            or validation.get("status") != "PASS"
            or validation.get("classifier_trained") is not False):
        raise ValueError("dataset handoff provenance or validation report is not approved")
    # This read-only check verifies pinned source hashes and reconstructed evidence.
    verified = verify_directory(dataset)
    if not verified.get("source_hashes_and_saved_evidence_verified"):
        raise ValueError("dataset evidence verification failed")
    frame = pd.read_csv(features_path, float_precision="round_trip")
    X, y, groups = validate_frame(frame)
    if len(frame) != 1170 or groups.nunique() != 30 or set(y) != set(CLASS_LABELS):
        raise ValueError("expected 1170 rows, 30 runs and all five V2 classes")
    if groups.value_counts().nunique() != 1 or groups.value_counts().iloc[0] != 39:
        raise ValueError("expected 39 windows per independent run")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("training features must be finite")
    return frame, X, y, groups, manifest, verified


def evaluate(X, y, groups):
    split_list = list(grouped_splits(X, y, groups))
    results = {}
    for name, prototype in build_models().items():
        oof = np.empty(len(y), dtype=object)
        folds = []
        for fold, train_idx, test_idx in split_list:
            estimator = sklearn.base.clone(prototype)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            predicted = estimator.predict(X.iloc[test_idx])
            oof[test_idx] = predicted
            folds.append({"fold": fold,
                "train_run_ids": sorted(groups.iloc[train_idx].unique().tolist()),
                "validation_run_ids": sorted(groups.iloc[test_idx].unique().tolist()),
                "run_id_overlap": False,
                "metrics": _summary(y.iloc[test_idx], predicted)})
        fold_scores = [fold["metrics"]["macro_f1"] for fold in folds]
        results[name] = {"configuration": _config(name), "folds": folds,
            "mean_fold_macro_f1": float(np.mean(fold_scores)),
            "std_fold_macro_f1": float(np.std(fold_scores)),
            "out_of_fold_metrics": _summary(y, oof)}
    selected_name = max(results, key=lambda name: results[name]["mean_fold_macro_f1"])
    selected = build_models()[selected_name]
    selected.fit(X, y)
    return results, selected_name, selected


def _config(name):
    if name == "random_forest":
        return {"n_estimators": 300, "max_features": "sqrt", "class_weight": "balanced",
                "random_state": SEED, "n_jobs": 1}
    return {"scaler": "StandardScaler", "kernel": "rbf", "C": 1.0,
            "gamma": "scale", "class_weight": "balanced", "probability": False}


def run(dataset=DEFAULT_DATASET, report_path=DEFAULT_REPORT, model_path=DEFAULT_MODEL):
    frame, X, y, groups, manifest, verification = load_training_handoff(dataset)
    results, selected_name, selected = evaluate(X, y, groups)
    report = {
        "report_version": "1.0", "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION, "source_commit": manifest["source_commit"],
        "dataset_provenance": {"data_kind": "synthetic_simulated_engine_conditions",
            "real_engine_measurements": False, "fault_labels_are_simulated_conditions": True,
            "producer_version": manifest["producer_version"]},
        "dataset": {"source_path": str(Path(dataset).as_posix()),
            "features_sha256": _sha256(Path(dataset) / "features.csv"), "rows": len(frame),
            "feature_count": len(FEATURE_COLUMNS), "metadata_columns_excluded": list(MANIFEST["metadata_columns"]),
            "label_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
            "unique_run_groups": int(groups.nunique()), "windows_per_run": 39,
            "handoff_verification": verification},
        "evaluation": {"method": "5-fold GroupKFold", "n_splits": N_SPLITS,
            "group_column": "run_id", "random_window_split_used": False,
            "run_ids_disjoint": True, "fold_class_coverage_required": "all five classes",
            "preprocessing_fitted_within_training_fold": True},
        "models": results,
        "selection": {"criterion": "highest mean grouped-fold macro F1",
            "selected_model": selected_name,
            "selected_mean_fold_macro_f1": results[selected_name]["mean_fold_macro_f1"]},
        "v1_comparison": load_v1_comparison(),
        "limitations": ["All examples and labels are synthetic simulator outputs, not real engine recordings.",
            "Grouped CV measures held-out simulator runs, not independent physical engines.",
            "V2 changes the feature contract and producer, so V1/V2 scores are descriptive only; no controlled improvement claim is made.",
            "Scores are model-specific and are not calibrated fault probabilities.",
            "MAP is an illustrative surrogate and CKP follows the declared synthetic trajectory."]}
    report_path, model_path = Path(report_path), Path(model_path)
    if report_path.exists() or model_path.exists():
        raise FileExistsError("refusing to overwrite existing V2 report or model")
    bundle = {"model": selected, "model_name": selected_name, "model_version": MODEL_VERSION,
        "feature_columns": list(FEATURE_COLUMNS), "class_labels": list(CLASS_LABELS),
        "schema_version": SCHEMA_VERSION, "source_commit": manifest["source_commit"],
        "dataset_sha256": report["dataset"]["features_sha256"], "configuration": results[selected_name]["configuration"],
        "training_provenance": report["dataset_provenance"],
        "inference_requirements": {"all_167_features": True, "all_quality_status": "good",
                                   "metadata_excluded": True, "scores_uncalibrated": True}}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report["artifacts"] = {"model_path": model_path.as_posix(), "model_sha256": _sha256(model_path)}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["report_sha256"] = _sha256(report_path)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def load_v1_comparison():
    path = Path("ml/evaluation/synthetic_conditions_136_groupkfold.json")
    if not path.exists():
        return {"status": "unavailable", "claim": "No V1 report found; comparison omitted."}
    report = json.loads(path.read_text(encoding="utf-8"))
    model = report.get("selection", {}).get("selected_model")
    metrics = report.get("models", {}).get(model, {}).get("out_of_fold_metrics", {})
    return {"status": "descriptive_only", "v1_model_version": report.get("model_version"),
            "v1_selected_model": model, "v1_grouped_cv_macro_f1": metrics.get("macro_f1"),
            "v1_grouped_cv_accuracy": metrics.get("accuracy"),
            "interpretation": "Different contracts/producers; not a controlled comparison and no improvement claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    report = run(args.dataset, args.report, args.model_output)
    print(json.dumps(report["selection"], indent=2))
    print("Report:", args.report)
    print("Model:", args.model_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
