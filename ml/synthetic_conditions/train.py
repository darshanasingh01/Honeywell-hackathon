"""Run leakage-safe grouped CV for Person B's synthetic engine features."""

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


META_COLUMNS = ("run_id", "window_id", "label", "t_start", "t_end")
CLASS_ORDER = ("normal", "knock", "misfire", "overheating", "valve_train_wear")
N_SPLITS = 5
RANDOM_STATE = 42
MODEL_VERSION = "synthetic-engine-condition-baseline-v1"
DEFAULT_REPORT = Path("ml/evaluation/synthetic_conditions_groupkfold.json")
DEFAULT_MODEL = Path("ml/saved_models/synthetic_conditions_baseline.joblib")


def _portable_path(value: str | Path | None) -> str | None:
    """Avoid recording machine-specific absolute paths in portable reports."""
    if value is None:
        return None
    path = Path(value)
    return path.name if path.is_absolute() else path.as_posix()


def prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Validate the handoff table and strictly exclude its five metadata fields."""
    missing = set(META_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"features table is missing metadata columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("features table is empty")
    if frame[list(META_COLUMNS)].isna().any().any():
        raise ValueError("metadata columns contain missing values")

    # This is the explicit model input contract: no labels, times, or group IDs.
    X = frame.drop(columns=list(META_COLUMNS))
    y = frame["label"].astype(str)
    groups = frame["run_id"].astype(str)
    if X.shape[1] != 74:
        raise ValueError(f"expected 74 numerical features after metadata exclusion; got {X.shape[1]}")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("all model feature columns must be numeric")
    values = X.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("model features contain missing or non-finite values")
    if set(y.unique()) != set(CLASS_ORDER):
        raise ValueError(f"expected classes {list(CLASS_ORDER)}, got {sorted(y.unique().tolist())}")
    labels_per_run = frame.assign(_label=y, _group=groups).groupby("_group")["_label"].nunique()
    if (labels_per_run != 1).any():
        raise ValueError("each run_id must map to exactly one condition label")
    if groups.nunique() < N_SPLITS:
        raise ValueError(f"GroupKFold requires at least {N_SPLITS} unique run_id groups")
    return X, y, groups


def build_models() -> dict[str, Any]:
    """Return fixed baselines; no hyperparameter search is performed."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "scaled_rbf_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
            )),
        ]),
    }


def grouped_splits(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """Yield GroupKFold indices and assert no run crosses each fold boundary."""
    splitter = GroupKFold(n_splits=N_SPLITS)
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        if train_groups & test_groups:
            raise RuntimeError("run_id leakage detected across a GroupKFold split")
        yield train_idx, test_idx


def _class_metrics(y_true, y_pred) -> dict[str, dict[str, float | int]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0,
    )
    return {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(CLASS_ORDER)
    }


def _metric_summary(y_true, y_pred) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).astype(int).tolist(),
        "class_order": list(CLASS_ORDER),
        "class_metrics": _class_metrics(y_true, y_pred),
    }


def _evaluate_one_model(name: str, estimator, X: pd.DataFrame, y: pd.Series,
                        groups: pd.Series) -> dict[str, Any]:
    oof = np.empty(len(y), dtype=object)
    fold_reports = []
    all_indices = []
    for fold, (train_idx, test_idx) in enumerate(grouped_splits(X, y, groups), start=1):
        train_y = y.iloc[train_idx]
        test_y = y.iloc[test_idx]
        test_groups = sorted(groups.iloc[test_idx].unique().tolist())
        train_groups = sorted(groups.iloc[train_idx].unique().tolist())
        missing_train = sorted(set(CLASS_ORDER) - set(train_y.unique()))
        missing_test = sorted(set(CLASS_ORDER) - set(test_y.unique()))

        # Every transformation (notably StandardScaler) is fitted by estimator.fit
        # using only this fold's training rows.
        estimator.fit(X.iloc[train_idx], train_y)
        predicted = estimator.predict(X.iloc[test_idx])
        oof[test_idx] = predicted
        all_indices.extend(test_idx.tolist())
        fold_reports.append({
            "fold": fold,
            "train_group_count": len(train_groups),
            "test_group_count": len(test_groups),
            "test_groups": test_groups,
            "train_class_coverage": {label: label not in missing_train for label in CLASS_ORDER},
            "test_class_coverage": {label: label not in missing_test for label in CLASS_ORDER},
            "missing_train_classes": missing_train,
            "missing_test_classes": missing_test,
            "metrics": _metric_summary(test_y, predicted),
        })

    if sorted(all_indices) != list(range(len(y))):
        raise RuntimeError("GroupKFold did not produce exactly one held-out prediction per row")
    fold_macro = [fold["metrics"]["macro_f1"] for fold in fold_reports]
    return {
        "hyperparameters": _model_hyperparameters(name),
        "folds": fold_reports,
        "mean_fold_macro_f1": float(np.mean(fold_macro)),
        "std_fold_macro_f1": float(np.std(fold_macro, ddof=0)),
        "out_of_fold_metrics": _metric_summary(y, oof),
    }


def _model_hyperparameters(name: str) -> dict[str, Any]:
    if name == "random_forest":
        return {
            "n_estimators": 300, "max_features": "sqrt", "class_weight": "balanced",
            "random_state": RANDOM_STATE, "n_jobs": 1,
        }
    return {"scaler": "StandardScaler", "kernel": "rbf", "C": 1.0,
            "gamma": "scale", "class_weight": "balanced", "probability": False}


def evaluate(frame: pd.DataFrame, *, source_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    X, y, groups = prepare_xy(frame)
    results = {
        name: _evaluate_one_model(name, estimator, X, y, groups)
        for name, estimator in build_models().items()
    }
    selected_name = max(results, key=lambda key: results[key]["mean_fold_macro_f1"])
    selected = build_models()[selected_name]
    selected.fit(X, y)
    provenance = {
        "data_origin": "Person B simulator feature-extraction pipeline",
        "data_kind": "synthetic_simulated_engine_conditions",
        "real_engine_measurements": False,
        "fault_labels_are_simulated_conditions": True,
    }
    report: dict[str, Any] = {
        "report_version": "1.0",
        "model_version": MODEL_VERSION,
        "dataset_provenance": provenance,
        "dataset": {
            "source_path": _portable_path(source_path),
            "rows": int(len(frame)),
            "feature_count": int(X.shape[1]),
            "metadata_columns_excluded": list(META_COLUMNS),
            "classes": list(CLASS_ORDER),
            "label_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "unique_run_groups": int(groups.nunique()),
            "missing_feature_values": int(X.isna().sum().sum()),
        },
        "evaluation": {
            "method": "GroupKFold",
            "n_splits": N_SPLITS,
            "group_column": "run_id",
            "random_window_split_used": False,
            "preprocessing_fitted_within_training_fold": True,
            "class_order": list(CLASS_ORDER),
        },
        "models": results,
        "selection": {
            "criterion": "highest mean grouped-CV fold macro F1",
            "selected_model": selected_name,
            "selected_mean_fold_macro_f1": results[selected_name]["mean_fold_macro_f1"],
        },
        "limitations": [
            "All training and evaluation examples come from a deterministic simulator; no real engine recordings are represented.",
            "The classes are simulated operating/fault conditions and do not establish real-world fault-detection accuracy.",
            "GroupKFold holds out run_id groups from the same simulator, not independent physical engines or vehicles.",
            "No CNN, random window-level split, or cross-dataset evaluation was performed.",
            "Any class score emitted at inference is a model score and must not be interpreted as a validated real-world fault probability.",
        ],
    }
    return report, {
        "model": selected,
        "model_name": selected_name,
        "model_version": MODEL_VERSION,
        "feature_columns": X.columns.tolist(),
        "class_labels": list(CLASS_ORDER),
        "hyperparameters": results[selected_name]["hyperparameters"],
        "training_provenance": provenance,
    }


def run(features_path: str | Path, report_path: str | Path = DEFAULT_REPORT,
        model_path: str | Path = DEFAULT_MODEL) -> dict[str, Any]:
    features_path = Path(features_path)
    frame = pd.read_csv(features_path)
    report, bundle = evaluate(frame, source_path=features_path)
    digest = hashlib.sha256(features_path.read_bytes()).hexdigest()
    report["dataset"]["sha256_features_csv"] = digest
    report["software"] = {"scikit_learn": sklearn.__version__}
    report_file, model_file = Path(report_path), Path(model_path)
    model_display_path = _portable_path(model_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file = model_file.resolve()
    joblib.dump(bundle, model_file)
    report["selected_model_artifact"] = model_display_path
    report_file.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Rows: {report['dataset']['rows']}; features: {report['dataset']['feature_count']}; run groups: {report['dataset']['unique_run_groups']}")
    for name, result in report["models"].items():
        print(f"{name}: mean fold macro-F1={result['mean_fold_macro_f1']:.6f}; OOF macro-F1={result['out_of_fold_metrics']['macro_f1']:.6f}; OOF accuracy={result['out_of_fold_metrics']['accuracy']:.6f}; OOF balanced accuracy={result['out_of_fold_metrics']['balanced_accuracy']:.6f}")
        for fold in result["folds"]:
            print(f"  fold {fold['fold']}: accuracy={fold['metrics']['accuracy']:.6f}; macro-F1={fold['metrics']['macro_f1']:.6f}; balanced_accuracy={fold['metrics']['balanced_accuracy']:.6f}; missing_test_classes={fold['missing_test_classes']}")
    print(f"Selected: {report['selection']['selected_model']}")
    print(f"Evaluation report: {report_file.resolve()}")
    print(f"Model artifact: {model_file}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Person B features.csv; read-only input")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    run(args.features, args.report, args.model_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
