"""Fit and group-evaluate an unlabeled vibration novelty reference model.

Manifest CSV required fields: path,sample_rate_hz,unit,group_id. Every group_id
must identify the engine/session/source recording that must stay in one fold.
No healthy or fault label is inferred from an unlabeled signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from features import extract_features, feature_names
from signals import load_signal, preprocess, segment_signal

ALERT_QUANTILE = 0.95


def rows_to_features(manifest: pd.DataFrame, root: Path, window_seconds: float, overlap: float, low_hz=None, high_hz=None):
    rows, groups = [], []
    for _, row in manifest.iterrows():
        signal_path = Path(row["path"])
        if not signal_path.is_absolute():
            signal_path = root / signal_path
        column = str(row["column"]) if "column" in row and pd.notna(row["column"]) else None
        signal = load_signal(signal_path, float(row["sample_rate_hz"]), str(row["unit"]), column)
        signal = preprocess(signal, low_hz=low_hz, high_hz=high_hz)
        for window in segment_signal(signal, window_seconds, overlap):
            rows.append(extract_features(window, signal.sample_rate_hz))
            groups.append(str(row["group_id"]))
    return pd.DataFrame(rows), np.asarray(groups)


def pipeline(seed: int):
    return make_pipeline(RobustScaler(), IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1))


def score(model, X):
    return -model.decision_function(X)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with path,sample_rate_hz,unit,group_id")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.0, help="Fractional overlap; keep 0 for strict non-overlap")
    parser.add_argument("--low-hz", type=float, help="Optional high-pass cutoff; validated against each sample rate")
    parser.add_argument("--high-hz", type=float, help="Optional low-pass cutoff; validated against each sample rate")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model-out", type=Path, default=Path("ml/saved_models/vibration_novelty.joblib"))
    parser.add_argument("--report-out", type=Path, default=Path("ml/evaluation/vibration_novelty.json"))
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = pd.read_csv(manifest_path)
    required = {"path", "sample_rate_hz", "unit", "group_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if manifest.empty or manifest[list(required)].isna().any().any():
        raise ValueError("Manifest must be nonempty with all required values present")
    if manifest["sample_rate_hz"].astype(float).nunique() != 1:
        raise ValueError("A single model requires one documented sample rate; harmonize data explicitly before training")
    if manifest["unit"].astype(str).str.strip().nunique() != 1:
        raise ValueError("A single model requires one documented physical unit; convert explicitly before training")
    X_frame, groups = rows_to_features(manifest, manifest_path.parent, args.window_seconds, args.overlap, args.low_hz, args.high_hz)
    X = X_frame[feature_names()].to_numpy(dtype=np.float64)
    unique_groups = np.unique(groups)
    if args.folds < 2 or args.folds > len(unique_groups):
        raise ValueError("folds must be between 2 and the number of unique group_id values")
    # Group holdout protects recordings and their derived windows from crossing a split.
    outer = GroupKFold(n_splits=args.folds)
    oof_scores = np.full(len(X), np.nan)
    oof_thresholds = np.full(len(X), np.nan)
    fold_summaries = []
    for fold_id, (train_idx, test_idx) in enumerate(outer.split(X, groups=groups), 1):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("Group leakage detected")
        inner_group_count = len(np.unique(groups[train_idx]))
        if inner_group_count < 2:
            raise ValueError("At least two training groups are required for threshold calibration")
        inner = GroupKFold(n_splits=min(4, inner_group_count))
        train_scores = []
        for inner_train, inner_cal in inner.split(X[train_idx], groups=groups[train_idx]):
            fit_rows, cal_rows = train_idx[inner_train], train_idx[inner_cal]
            detector = pipeline(1000 * fold_id + len(train_scores))
            detector.fit(X[fit_rows])
            train_scores.extend(score(detector, X[cal_rows]).tolist())
        threshold = float(np.quantile(train_scores, ALERT_QUANTILE))
        fold_model = pipeline(2000 + fold_id)
        fold_model.fit(X[train_idx])
        heldout_scores = score(fold_model, X[test_idx])
        oof_scores[test_idx] = heldout_scores
        oof_thresholds[test_idx] = threshold
        fold_summaries.append({
            "fold": fold_id,
            "train_groups": int(len(np.unique(groups[train_idx]))),
            "test_groups": int(len(np.unique(groups[test_idx]))),
            "train_windows": int(len(train_idx)),
            "test_windows": int(len(test_idx)),
            "training_only_calibrated_threshold": threshold,
            "heldout_alert_rate": float(np.mean(heldout_scores > threshold)),
            "heldout_score_median": float(np.median(heldout_scores)),
            "heldout_score_p95": float(np.quantile(heldout_scores, .95)),
        })
    if np.isnan(oof_scores).any():
        raise RuntimeError("Not every window received an out-of-fold score")
    # Cross-fitted training-group scores calibrate the deployment cutoff.
    inner = GroupKFold(n_splits=min(5, len(unique_groups)))
    calibration_scores = []
    for fold_id, (fit_idx, cal_idx) in enumerate(inner.split(X, groups=groups)):
        detector = pipeline(9000 + fold_id).fit(X[fit_idx])
        calibration_scores.extend(score(detector, X[cal_idx]).tolist())
    threshold = float(np.quantile(calibration_scores, ALERT_QUANTILE))
    final_model = pipeline(10000).fit(X)
    bundle = {
        "model_version": "engineguard-vibration-novelty-v1",
        "model": final_model,
        "threshold": threshold,
        "feature_names": feature_names(),
        "window_seconds": args.window_seconds,
        "overlap": args.overlap,
        "sample_rate_hz": float(manifest["sample_rate_hz"].iloc[0]),
        "unit": str(manifest["unit"].iloc[0]),
        "preprocessing": {"low_hz": args.low_hz, "high_hz": args.high_hz},
        "threshold_quantile": ALERT_QUANTILE,
        "calibration": "cross-fitted GroupKFold reference scores; training groups only",
        "limitations": ["Reference recordings are not automatically healthy.", "Novelty is not fault diagnosis."],
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)
    report = {
        "task": "unlabeled vibration novelty detection; not verified fault diagnosis",
        "recordings": int(len(manifest)),
        "groups": int(len(unique_groups)),
        "windows": int(len(X)),
        "window_seconds": args.window_seconds,
        "overlap": args.overlap,
        "units": sorted(manifest["unit"].astype(str).unique().tolist()),
        "sample_rates_hz": sorted(manifest["sample_rate_hz"].astype(float).unique().tolist()),
        "outer_split": "GroupKFold by manifest group_id; all windows from each source group stay in one fold",
        "threshold_quantile": ALERT_QUANTILE,
        "outer_folds": fold_summaries,
        "oof_reference_alert_rate": float(np.mean(oof_scores > oof_thresholds)),
        "oof_score_distribution": {"mean": float(np.mean(oof_scores)), "std": float(np.std(oof_scores)), "median": float(np.median(oof_scores)), "p95": float(np.quantile(oof_scores, .95))},
        "fault_metrics": None,
        "caveat": "Unlabeled reference data do not establish healthy/fault status; alert rate is not a false-fault rate. No fault labels or synthetic fault performance are claimed.",
        "model_artifact": str(args.model_out),
    }
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("recordings", "groups", "windows", "oof_reference_alert_rate", "model_artifact")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
