"""Train and evaluate the HL-CEAD microphone RPM classification baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract_features
from preprocess import load_audio, split_audio

RPM_CLASSES = ("1000", "1500", "2000")


def discover_recordings(root: Path) -> list[dict[str, str | Path]]:
    """Read metadata first so split validity is checked before feature extraction."""
    paths = sorted(root.rglob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"No WAV files found under {root}")
    records = []
    for path in paths:
        relative = path.relative_to(root)
        if len(relative.parts) != 4:
            raise ValueError(f"Unexpected HL-CEAD path layout: {relative}")
        manufacturer, model, rpm, _filename = relative.parts
        if rpm not in RPM_CLASSES:
            raise ValueError(f"Unexpected RPM directory {rpm!r}: {relative}")
        records.append({
            "path": path,
            "label": rpm,
            "car_id": f"{manufacturer}/{model}/{path.stem}",
            "vehicle_model": f"{manufacturer}/{model}",
            "relative_path": relative.as_posix(),
        })
    return records


def validate_grouped_splits(labels: np.ndarray, groups: np.ndarray, n_splits: int):
    """Return grouped folds after asserting complete RPM coverage and no leakage."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    expected = set(RPM_CLASSES)
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        train_groups, test_groups = set(groups[train_idx]), set(groups[test_idx])
        if train_groups & test_groups:
            raise ValueError(f"Fold {fold_number} leaks car groups across train and test")
        missing_train = expected - set(labels[train_idx])
        missing_test = expected - set(labels[test_idx])
        if missing_train or missing_test:
            raise ValueError(
                f"Fold {fold_number} lacks RPM classes: train={sorted(missing_train)}, "
                f"test={sorted(missing_test)}. Reduce --folds or inspect the dataset."
            )
    return folds


def extract_recording_features(records, sample_rate: int, seconds: float) -> np.ndarray:
    vectors = []
    for record in records:
        audio = load_audio(record["path"], sample_rate)
        chunks = split_audio(audio, sample_rate, seconds)
        if not chunks:
            raise ValueError(f"Recording shorter than {seconds:g}s: {record['relative_path']}")
        # All chunks from a WAV are combined here, before CV. No segments from
        # one source recording can be distributed between train and test.
        segment_features = [extract_features(chunk, sample_rate) for chunk in chunks]
        vectors.append(np.mean(segment_features, axis=0))
    return np.stack(vectors)


def make_estimator():
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="HL-CEAD root containing Manufacturer/Model/RPM/*.wav")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model-out", type=Path, default=Path("ml/saved_models/acoustic_hlcead_rpm.joblib"))
    parser.add_argument("--report-out", type=Path, default=Path("ml/evaluation/acoustic_hlcead_rpm.json"))
    args = parser.parse_args()

    root = args.dataset.expanduser().resolve()
    records = discover_recordings(root)
    labels = np.asarray([record["label"] for record in records])
    groups = np.asarray([record["car_id"] for record in records])
    if args.folds < 2 or args.folds > len(np.unique(groups)):
        raise ValueError("--folds must be between 2 and the number of unique cars")

    # Validate the exact metadata-based splits before spending time on audio features.
    folds = validate_grouped_splits(labels, groups, args.folds)
    X = extract_recording_features(records, args.sample_rate, args.segment_seconds)

    oof_predictions = np.empty_like(labels)
    fold_metrics = []
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        estimator = make_estimator()
        estimator.fit(X[train_idx], labels[train_idx])
        predictions = estimator.predict(X[test_idx])
        oof_predictions[test_idx] = predictions
        fold_metrics.append({
            "fold": fold_number,
            "train_cars": int(len(np.unique(groups[train_idx]))),
            "test_cars": int(len(np.unique(groups[test_idx]))),
            "train_rpm_counts": {rpm: int(np.sum(labels[train_idx] == rpm)) for rpm in RPM_CLASSES},
            "test_rpm_counts": {rpm: int(np.sum(labels[test_idx] == rpm)) for rpm in RPM_CLASSES},
            "accuracy": float(accuracy_score(labels[test_idx], predictions)),
            "macro_f1": float(f1_score(labels[test_idx], predictions, labels=RPM_CLASSES, average="macro", zero_division=0)),
        })

    final_model = make_estimator().fit(X, labels)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "sample_rate": args.sample_rate,
        "segment_seconds": args.segment_seconds,
        "classes": list(RPM_CLASSES),
        "feature_version": 1,
        "target": "engine_rpm",
    }, args.model_out)

    example_index = int(folds[0][1][0])
    report = {
        "dataset": "HL-CEAD",
        "target": "engine_rpm",
        "recordings": len(records),
        "cars": int(len(np.unique(groups))),
        "vehicle_model_labels": int(len({record['vehicle_model'] for record in records})),
        "rpm_recording_counts": {rpm: int(np.sum(labels == rpm)) for rpm in RPM_CLASSES},
        "evaluation": "StratifiedGroupKFold; 5 folds; group=manufacturer/model/recording stem (car); one feature vector per source WAV",
        "folds": fold_metrics,
        "oof_accuracy": float(accuracy_score(labels, oof_predictions)),
        "oof_macro_f1": float(f1_score(labels, oof_predictions, labels=RPM_CLASSES, average="macro", zero_division=0)),
        "oof_confusion_matrix_labels": list(RPM_CLASSES),
        "oof_confusion_matrix": confusion_matrix(labels, oof_predictions, labels=RPM_CLASSES).tolist(),
        "held_out_example": {
            "relative_path": records[example_index]["relative_path"],
            "car_id": records[example_index]["car_id"],
            "true_rpm": labels[example_index],
            "oof_predicted_rpm": oof_predictions[example_index],
            "fold": 1,
        },
        "note": "RPM estimation baseline only. HL-CEAD contains no fault labels; this does not identify engine faults.",
    }
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Recordings: {len(records)}; cars: {len(np.unique(groups))}; model labels: {report['vehicle_model_labels']}")
    print(f"RPM counts: {report['rpm_recording_counts']}")
    print(f"OOF accuracy: {report['oof_accuracy']:.4f}; macro-F1: {report['oof_macro_f1']:.4f}")
    print(f"Confusion matrix (rows=true, columns=predicted; {RPM_CLASSES}): {report['oof_confusion_matrix']}")
    print(f"Held-out example: {report['held_out_example']}")
    print(f"Saved model: {args.model_out}; report: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
