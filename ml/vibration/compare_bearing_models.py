"""Compare logistic regression and a balanced random forest on fixed NUST CV folds.

This evaluation-only script does not fit or save a final model. Each CSV is one
recording, features are Channel 3 time-domain statistics, and preprocessing is
inside each training fold. Evaluation splits are stratified by label at the
recording level; the dataset has no verified engine/session IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .bearing_condition import FEATURE_NAMES, FAULTY_LABEL, HEALTHY_LABEL, LABELS, discover_recordings, extract_time_domain_features
    from .train_bearing_condition import stratified_recording_folds
except ImportError:  # Support direct script execution from the repository root.
    from bearing_condition import FEATURE_NAMES, FAULTY_LABEL, HEALTHY_LABEL, LABELS, discover_recordings, extract_time_domain_features
    from train_bearing_condition import stratified_recording_folds


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_label_order": list(LABELS),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
        "faulty_bearing_precision": float(precision_score(y_true, y_pred, pos_label=FAULTY_LABEL, zero_division=0)),
        "faulty_bearing_recall": float(recall_score(y_true, y_pred, pos_label=FAULTY_LABEL, zero_division=0)),
        "healthy_bearing_false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
    }


def _candidate(name: str, seed: int):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed))
    if name == "balanced_random_forest":
        return make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        ))
    raise ValueError(f"Unknown candidate: {name}")


def evaluate(records: list, *, folds: int, seed: int) -> dict:
    y = np.asarray([r.label for r in records], dtype=object)
    X = pd.DataFrame([extract_time_domain_features(r.samples) for r in records], columns=FEATURE_NAMES)
    split_indices = list(stratified_recording_folds(y, folds=folds, seed=seed))
    predictions = {name: np.full(y.shape, None, dtype=object) for name in ("logistic_regression", "balanced_random_forest")}
    fold_results = {name: [] for name in predictions}

    for fold_id, (train_idx, test_idx) in enumerate(split_indices, start=1):
        for name in predictions:
            model = _candidate(name, seed + fold_id)
            model.fit(X.iloc[train_idx], y[train_idx])
            fold_pred = model.predict(X.iloc[test_idx])
            predictions[name][test_idx] = fold_pred
            fold_results[name].append({"fold": fold_id, "recordings": int(len(test_idx)), **_metrics(y[test_idx], fold_pred)})

    tn = int(np.sum(y == HEALTHY_LABEL))
    tp = int(np.sum(y == FAULTY_LABEL))
    return {
        "recordings": int(len(records)),
        "class_counts": {HEALTHY_LABEL: tn, FAULTY_LABEL: tp},
        "split": f"{folds}-fold StratifiedKFold, seed {seed}; identical indices for both candidates; one complete CSV per row",
        "grouping_limit": "Exact byte duplicates removed before splitting. No engine or session IDs exist, so this is recording-level, not engine/session-grouped CV.",
        "feature_domain": "Channel 3 time-domain sample-value statistics only; no sample-rate-dependent features",
        "candidates": {
            name: {"oof_metrics": _metrics(y, pred), "folds": fold_results[name]}
            for name, pred in predictions.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("ml/datasets/vibration/nust/extracted"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-out", type=Path, default=Path("ml/evaluation/vibration_nust_model_comparison.json"))
    args = parser.parse_args()

    records, dedupe = discover_recordings(args.dataset)
    may27 = [record for record in records if record.recording_date == "2022May27"]
    if not may27 or {record.label for record in may27} != set(LABELS):
        raise ValueError("Evaluation B requires both bearing labels among May 27 recordings")
    report = {
        "task": "evaluation-only NUST bearing-condition candidate comparison",
        "candidates": {
            "logistic_regression": "StandardScaler + LogisticRegression(solver=liblinear, max_iter=2000); existing baseline configuration",
            "balanced_random_forest": "StandardScaler + RandomForestClassifier(n_estimators=300, class_weight=balanced); fixed default candidate, not tuned",
        },
        "preprocessing": "Feature extraction is recording-local; StandardScaler is fit inside each training fold.",
        "discovery": {**dedupe, "unique_recordings": len(records)},
        "evaluations": {
            "A_all_unique_recordings": evaluate(records, folds=args.folds, seed=args.seed),
            "B_may_27_only": evaluate(may27, folds=args.folds, seed=args.seed),
        },
        "limitations": [
            "May 27 healthy/faulty subset is a date-restricted sensitivity analysis, not independent-session validation.",
            "Dataset represents one engine and has no usable session IDs; recording-level folds cannot establish unseen-engine or unseen-session performance.",
            "Class labels represent the dataset's healthy_bearing/faulty_bearing categories; mechanism and severity are unspecified.",
            "No threshold or hyperparameter selection used held-out fold results; no fitted model is saved.",
        ],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.report_out),
        "eval_A": {name: value["oof_metrics"] for name, value in report["evaluations"]["A_all_unique_recordings"]["candidates"].items()},
        "eval_B": {name: value["oof_metrics"] for name, value in report["evaluations"]["B_may_27_only"]["candidates"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
