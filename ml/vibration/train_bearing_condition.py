"""Train and recording-level cross-validate the NUST bearing-condition baseline.

This is an exploratory binary bearing-condition classifier, not a knock or
combustion fault-type classifier. It uses one time-domain feature vector per
CSV and does not assume a sample rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .bearing_condition import (CHANNEL, FEATURE_NAMES, FAULTY_LABEL, HEALTHY_LABEL,
                                    LABELS, MODEL_VERSION, discover_recordings,
                                    extract_time_domain_features)
except ImportError:  # Support `python ml/vibration/train_bearing_condition.py`.
    from bearing_condition import (CHANNEL, FEATURE_NAMES, FAULTY_LABEL, HEALTHY_LABEL,
                                   LABELS, MODEL_VERSION, discover_recordings,
                                   extract_time_domain_features)


def _new_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed),
    )


def stratified_recording_folds(labels: np.ndarray, *, folds: int, seed: int):
    """Yield recording indices; each CSV contributes exactly one row to one fold."""
    labels = np.asarray(labels, dtype=object)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    counts = {label: int(np.sum(labels == label)) for label in LABELS}
    if min(counts.values()) < folds:
        raise ValueError("each class must have at least one recording per stratified fold")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    yield from splitter.split(np.zeros(labels.size), labels)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    per_precision, per_recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    return {
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_label_order": list(LABELS),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
        "faulty_bearing_precision": float(per_precision[1]),
        "faulty_bearing_recall": float(per_recall[1]),
    }


def _evaluate(name: str, records: list, *, folds: int, seed: int) -> dict:
    y = np.asarray([r.label for r in records], dtype=object)
    counts = {label: int(np.sum(y == label)) for label in LABELS}
    minority = min(counts.values())
    if minority < folds:
        raise ValueError(f"Evaluation {name} has only {minority} examples in its smaller class; cannot run {folds}-fold stratified CV")
    X = pd.DataFrame([extract_time_domain_features(r.samples) for r in records], columns=FEATURE_NAMES)
    oof = np.full(y.shape, None, dtype=object)
    fold_metrics = []
    for fold_id, (train_idx, test_idx) in enumerate(stratified_recording_folds(y, folds=folds, seed=seed), 1):
        model = _new_model(seed + fold_id)
        model.fit(X.iloc[train_idx], y[train_idx])
        fold_pred = model.predict(X.iloc[test_idx])
        oof[test_idx] = fold_pred
        fold_metrics.append({"fold": fold_id, "recordings": int(len(test_idx)), **_metrics(y[test_idx], fold_pred)})
    if any(value is None for value in oof):
        raise RuntimeError(f"Evaluation {name} did not produce one OOF prediction for each recording")
    oof_metrics = _metrics(y, oof)
    majority = max(LABELS, key=lambda label: counts[label])
    baseline_pred = np.full(y.shape, majority, dtype=object)
    majority_metrics = _metrics(y, baseline_pred)
    metric_names = ("balanced_accuracy", "macro_f1", "faulty_bearing_precision", "faulty_bearing_recall")
    variability = {
        metric: {
            "mean": float(np.mean([fold[metric] for fold in fold_metrics])),
            "std": float(np.std([fold[metric] for fold in fold_metrics], ddof=1)) if len(fold_metrics) > 1 else 0.0,
        }
        for metric in metric_names
    }
    return {
        "recordings": int(len(records)),
        "class_counts": counts,
        "split": f"{folds}-fold StratifiedKFold at recording level; one complete CSV per row; no windows",
        "date_confound_warning": "Healthy recordings all originate from May 27; recording-held-out folds may share acquisition dates.",
        "oof_metrics": oof_metrics,
        "fold_metric_mean_and_std": variability,
        "folds": fold_metrics,
        "majority_class_baseline": {"predicted_class": majority, **majority_metrics},
        "recording_dates": sorted({r.recording_date for r in records if r.recording_date}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path,
                        default=Path("ml/datasets/vibration/nust/extracted"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=Path,
                        default=Path("ml/saved_models/vibration_nust_bearing_condition.joblib"))
    parser.add_argument("--report-out", type=Path,
                        default=Path("ml/evaluation/vibration_nust_bearing_condition.json"))
    args = parser.parse_args()
    records, dedupe = discover_recordings(args.dataset)
    all_records = records
    may27_records = [r for r in records if r.recording_date == "2022May27"]
    if not may27_records:
        raise ValueError("Evaluation B requires healthy and faulty recordings dated May 27")
    evaluations = {
        "A_all_unique_recordings": _evaluate("A", all_records, folds=args.folds, seed=args.seed),
        "B_may_27_recordings_only": _evaluate("B", may27_records, folds=args.folds, seed=args.seed),
    }

    # The final artifact is fitted only after both predeclared CV evaluations finish.
    X_all = pd.DataFrame([extract_time_domain_features(r.samples) for r in all_records], columns=FEATURE_NAMES)
    y_all = np.asarray([r.label for r in all_records], dtype=object)
    final_model = _new_model(args.seed).fit(X_all, y_all)
    bundle = {
        "model_version": MODEL_VERSION,
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "input_channel": CHANNEL,
        "input_unit": "g (as documented by the dataset paper)",
        "class_labels": list(LABELS),
        "probabilities_calibrated": False,
        "sample_rate_hz": None,
        "training_scope": "All byte-deduplicated NUST v5 small-archive recordings; one feature vector per recording.",
        "limitations": [
            "Dataset represents one engine; unseen-engine generalization is not measured.",
            "Healthy recordings are all dated May 27; recording-held-out CV can remain date-confounded.",
            "CSV timestamps are irregular and do not verify the paper-reported 3200 Hz rate.",
            "Only Channel 3 is used; no X/Y/Z channel mapping is asserted.",
            "Labels are only the dataset's healthy_bearing and faulty_bearing categories; fault mechanism and severity are unspecified.",
            "Probabilities are model scores and are not calibrated probabilities.",
            "Time-domain features summarize exported sample values; they are not duration-weighted because the timestamp stream is irregular.",
        ],
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)
    report = {
        "task": "exploratory supervised healthy-vs-faulty engine journal-bearing condition classification",
        "feature_domain": "recording-level Channel 3 sample-value statistics; no sampling-rate-dependent features",
        "channel": CHANNEL,
        "class_labels": list(LABELS),
        "dataset": str(args.dataset),
        "discovery": {**dedupe, "unique_recordings_after_byte_deduplication": len(records)},
        "evaluations": evaluations,
        "evaluation_agreement": {
            "balanced_accuracy_absolute_difference": abs(evaluations["A_all_unique_recordings"]["oof_metrics"]["balanced_accuracy"] - evaluations["B_may_27_recordings_only"]["oof_metrics"]["balanced_accuracy"]),
            "macro_f1_absolute_difference": abs(evaluations["A_all_unique_recordings"]["oof_metrics"]["macro_f1"] - evaluations["B_may_27_recordings_only"]["oof_metrics"]["macro_f1"]),
        },
        "model_artifact": str(args.model_out),
        "model_version": MODEL_VERSION,
        "limitations": bundle["limitations"],
    }
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"model_artifact": str(args.model_out), "report": str(args.report_out),
                      "eval_A": evaluations["A_all_unique_recordings"]["oof_metrics"],
                      "eval_B": evaluations["B_may_27_recordings_only"]["oof_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
