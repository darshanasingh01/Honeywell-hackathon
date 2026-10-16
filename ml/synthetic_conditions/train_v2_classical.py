"""Train separate synthetic-only V2 SVM and softmax baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .schema_v2 import FEATURE_COLUMNS, MANIFEST, SCHEMA_VERSION
from .train_v2 import (CLASS_LABELS, DEFAULT_DATASET, N_SPLITS, SEED,
                       _summary, grouped_splits, load_training_handoff)

SVM_VERSION = "synthetic-conditions-167-v2-svm-v1"
SOFTMAX_VERSION = "synthetic-conditions-167-v2-softmax-v1"
DEFAULT_REPORT = Path("ml/evaluation/synthetic_conditions_167_v2_classical_groupkfold.json")
DEFAULT_SVM_MODEL = Path("ml/saved_models/synthetic_conditions_167_v2_svm.joblib")
DEFAULT_SOFTMAX_MODEL = Path("ml/saved_models/synthetic_conditions_167_v2_softmax.joblib")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_models() -> dict[str, Pipeline]:
    """Return fresh pipelines; each scaler is fitted only by a training fold."""
    return {
        "scaled_rbf_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                               class_weight="balanced", probability=False)),
        ]),
        "standardized_softmax": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                # lbfgs uses multinomial softmax for multiclass targets.
                solver="lbfgs", C=1.0,
                class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]),
    }


def _config(name: str) -> dict[str, Any]:
    if name == "scaled_rbf_svm":
        return {"scaler": "StandardScaler", "kernel": "rbf", "C": 1.0,
                "gamma": "scale", "class_weight": "balanced",
                "probability": False}
    return {"scaler": "StandardScaler", "solver": "lbfgs",
            "multi_class": "multinomial", "C": 1.0,
            "class_weight": "balanced", "max_iter": 2000,
            "random_state": SEED}


def evaluate(X, y, groups) -> tuple[dict[str, Any], dict[str, Pipeline]]:
    split_list = list(grouped_splits(X, y, groups))
    results: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    for name, prototype in build_models().items():
        oof = np.empty(len(y), dtype=object)
        folds = []
        for fold, train_idx, test_idx in split_list:
            estimator = clone(prototype)
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            predicted = estimator.predict(X.iloc[test_idx])
            oof[test_idx] = predicted
            folds.append({
                "fold": fold,
                "train_run_ids": sorted(groups.iloc[train_idx].unique().tolist()),
                "validation_run_ids": sorted(groups.iloc[test_idx].unique().tolist()),
                "run_id_overlap": False,
                "metrics": _summary(y.iloc[test_idx], predicted),
            })
        fold_scores = [fold["metrics"]["macro_f1"] for fold in folds]
        final = clone(prototype)
        final.fit(X, y)
        fitted[name] = final
        results[name] = {
            "configuration": _config(name),
            "folds": folds,
            "mean_fold_macro_f1": float(np.mean(fold_scores)),
            "std_fold_macro_f1": float(np.std(fold_scores)),
            "out_of_fold_metrics": _summary(y, oof),
        }
    return results, fitted


def _bundle(model: Pipeline, name: str, version: str, dataset_sha256: str) -> dict[str, Any]:
    return {
        "model": model,
        "model_name": name,
        "model_version": version,
        "schema_version": SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "class_labels": list(CLASS_LABELS),
        "dataset_sha256": dataset_sha256,
        "training_provenance": {
            "data_kind": "synthetic_simulated_engine_conditions",
            "real_engine_measurements": False,
            "fault_labels_are_simulated_conditions": True,
            "training_module": "ml.synthetic_conditions.train_v2_classical",
        },
        "inference_requirements": {
            "all_167_features": True,
            "metadata_excluded": True,
            "scores_uncalibrated": True,
            "class_order": list(CLASS_LABELS),
        },
    }


def run(dataset=DEFAULT_DATASET, report_path=DEFAULT_REPORT,
        svm_model_path=DEFAULT_SVM_MODEL, softmax_model_path=DEFAULT_SOFTMAX_MODEL):
    import joblib

    report_path, svm_model_path, softmax_model_path = map(Path,
        (report_path, svm_model_path, softmax_model_path))
    for path in (report_path, svm_model_path, softmax_model_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    frame, X, y, groups, manifest, verification = load_training_handoff(dataset)
    results, fitted = evaluate(X, y, groups)
    features_path = Path(dataset) / "features.csv"
    dataset_sha = _sha256(features_path)
    report = {
        "report_version": "1.0",
        "report_kind": "synthetic_v2_classical_models",
        "schema_version": SCHEMA_VERSION,
        "model_versions": {"scaled_rbf_svm": SVM_VERSION,
                            "standardized_softmax": SOFTMAX_VERSION},
        "source_commit": manifest["source_commit"],
        "dataset_provenance": {
            "data_kind": "synthetic_simulated_engine_conditions",
            "real_engine_measurements": False,
            "fault_labels_are_simulated_conditions": True,
            "producer_version": manifest["producer_version"],
        },
        "dataset": {"source_path": str(Path(dataset).as_posix()),
            "features_sha256": dataset_sha, "rows": len(frame),
            "feature_count": len(FEATURE_COLUMNS), "unique_run_groups": int(groups.nunique()),
            "label_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
            "handoff_verification": verification},
        "evaluation": {"method": "5-fold GroupKFold", "n_splits": N_SPLITS,
            "group_column": "run_id", "run_ids_disjoint": True,
            "preprocessing_fitted_within_training_fold": True,
            "class_order": list(CLASS_LABELS)},
        "models": results,
        "artifacts": {"svm_path": svm_model_path.as_posix(),
            "softmax_path": softmax_model_path.as_posix()},
        "limitations": ["Training and evaluation use synthetic simulator data only.",
            "SVM decision scores are not probabilities.",
            "Softmax outputs are model probabilities but are not necessarily calibrated.",
            "Grouped CV measures held-out simulator runs, not independent physical engines."],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    svm_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_bundle(fitted["scaled_rbf_svm"], "scaled_rbf_svm", SVM_VERSION, dataset_sha), svm_model_path)
    joblib.dump(_bundle(fitted["standardized_softmax"], "standardized_softmax", SOFTMAX_VERSION, dataset_sha), softmax_model_path)
    report["artifacts"]["svm_sha256"] = _sha256(svm_model_path)
    report["artifacts"]["softmax_sha256"] = _sha256(softmax_model_path)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["artifacts"]["report_sha256"] = _sha256(report_path)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--svm-model-output", type=Path, default=DEFAULT_SVM_MODEL)
    parser.add_argument("--softmax-model-output", type=Path, default=DEFAULT_SOFTMAX_MODEL)
    args = parser.parse_args()
    report = run(args.dataset, args.report, args.svm_model_output, args.softmax_model_output)
    print(json.dumps({name: value["mean_fold_macro_f1"]
                      for name, value in report["models"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
