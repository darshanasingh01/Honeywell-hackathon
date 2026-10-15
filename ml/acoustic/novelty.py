"""RPM-conditioned acoustic reference novelty detector for HL-CEAD.

The source recordings have no verified healthy/fault labels. Scores here mean
"different from the HL-CEAD reference distribution at this RPM", never fault.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features
from preprocess import load_audio, split_audio
from train import RPM_CLASSES, discover_recordings, extract_recording_features, make_estimator

FEATURE_CONFIG = {
    "sample_rate": 22050,
    "segment_seconds": 3.0,
    "features": "20 MFCC + spectral centroid/bandwidth/rolloff + RMS + ZCR; per-segment mean/std; recording mean",
    "version": 1,
}
TRANSIENT_FEATURE_CONFIG = {
    **FEATURE_CONFIG,
    "features": FEATURE_CONFIG["features"] + "; transient max/95th spectral flux, short-time energy peak/median, crest factor, and impulsiveness (standardized per training fold)",
    "version": 2,
}
EXPECTED_REFERENCE_ALERT_RATE = 0.05


def detector_pipeline(seed: int):
    return make_pipeline(
        StandardScaler(),
        IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        ),
    )


def novelty_score(detector, matrix: np.ndarray) -> np.ndarray:
    """Higher values indicate lower conformity to the fitted reference set."""
    return -detector.decision_function(matrix)


def crossfit_thresholds(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int = 4,
    seed: int = 42,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Calibrate per-RPM cutoffs using only group-held-out training vehicles."""
    if folds < 2 or folds > len(np.unique(groups)):
        raise ValueError("Not enough training cars for grouped threshold calibration")
    splitter = GroupKFold(n_splits=folds)
    reference_scores = {rpm: [] for rpm in RPM_CLASSES}
    for inner_train, inner_cal in splitter.split(X, labels, groups):
        for rpm_index, rpm in enumerate(RPM_CLASSES):
            fit_idx = inner_train[labels[inner_train] == rpm]
            cal_idx = inner_cal[labels[inner_cal] == rpm]
            if len(fit_idx) < 8 or len(cal_idx) == 0:
                raise ValueError(f"Insufficient {rpm} RPM cars for grouped calibration")
            detector = detector_pipeline(seed + rpm_index)
            detector.fit(X[fit_idx])
            reference_scores[rpm].extend(novelty_score(detector, X[cal_idx]).tolist())

    thresholds = {
        rpm: float(np.quantile(reference_scores[rpm], 1.0 - EXPECTED_REFERENCE_ALERT_RATE))
        for rpm in RPM_CLASSES
    }
    return thresholds, reference_scores


def fit_rpm_detectors(X: np.ndarray, labels: np.ndarray, seed: int = 142):
    """Fit a separate scaler and Isolation Forest per RPM class."""
    models = {}
    counts = {}
    for rpm_index, rpm in enumerate(RPM_CLASSES):
        indices = np.flatnonzero(labels == rpm)
        if len(indices) < 8:
            raise ValueError(f"Too few training recordings at {rpm} RPM: {len(indices)}")
        model = detector_pipeline(seed + rpm_index)
        model.fit(X[indices])
        models[rpm] = model
        counts[rpm] = int(len(indices))
    return models, counts


def score_distribution(scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    return {
        "count": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "median": float(np.median(scores)),
        "p95": float(np.quantile(scores, 0.95)),
        "alert_rate": float(np.mean(scores > threshold)),
    }


def perturb_audio(audio: np.ndarray, sample_rate: int, kind: str) -> np.ndarray:
    """Create explicitly artificial stress examples; these are not engine faults."""
    out = audio.astype(np.float32, copy=True)
    rms = float(np.sqrt(np.mean(np.square(out)) + 1e-12))
    duration = len(out) / sample_rate
    if kind == "synthetic_impulsive_knocks":
        width = max(1, int(0.025 * sample_rate))
        t = np.arange(width) / sample_rate
        pulse = 1.5 * rms * np.exp(-t / 0.008) * np.sin(2 * np.pi * 1300 * t)
        for center_sec in np.arange(0.75, duration, 1.7):
            start = int(center_sec * sample_rate)
            end = min(len(out), start + width)
            out[start:end] += pulse[: end - start]
    elif kind == "synthetic_periodic_clicks":
        width = max(1, int(0.004 * sample_rate))
        for center_sec in np.arange(0.25, duration, 0.22):
            start = int(center_sec * sample_rate)
            end = min(len(out), start + width)
            out[start:end] += 2.0 * rms
    elif kind == "synthetic_amplitude_modulation":
        t = np.arange(len(out)) / sample_rate
        out *= (1.0 + 0.65 * np.sin(2 * np.pi * 7.0 * t)).astype(np.float32)
    else:
        raise ValueError(f"Unknown perturbation: {kind}")
    return out


def feature_from_audio(audio: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    chunks = split_audio(audio, sample_rate, seconds)
    if not chunks:
        raise ValueError("Audio is shorter than one analysis segment")
    return np.mean([extract_features(chunk, sample_rate) for chunk in chunks], axis=0)


def transient_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Recording-level transient summaries preserve brief peaks lost by means."""
    frame = max(256, int(0.020 * sample_rate))
    hop = max(64, int(0.005 * sample_rate))
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame)[::hop]
    energy = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    peak = np.max(np.abs(frames), axis=1)
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame), axis=1))
    normalized = spectrum / (np.sum(spectrum, axis=1, keepdims=True) + 1e-12)
    flux = np.sqrt(np.sum(np.maximum(np.diff(normalized, axis=0), 0.0) ** 2, axis=1))
    centered = audio.astype(np.float64) - float(np.mean(audio))
    rms = np.sqrt(np.mean(centered ** 2) + 1e-12)
    kurtosis = np.mean(centered ** 4) / (rms ** 4 + 1e-12)
    return np.asarray([
        np.quantile(flux, .95) if flux.size else 0.0,
        np.max(flux) if flux.size else 0.0,
        np.quantile(energy, .95) / (np.median(energy) + 1e-12),
        np.max(energy) / (np.median(energy) + 1e-12),
        np.max(np.abs(audio)) / (np.sqrt(np.mean(audio ** 2) + 1e-12)),
        kurtosis,
        np.quantile(peak, .99) / (np.median(energy) + 1e-12),
    ], dtype=np.float32)


def recording_features(record, sample_rate: int, seconds: float, feature_set: str) -> np.ndarray:
    base = extract_recording_features([record], sample_rate, seconds)[0]
    if feature_set == "baseline":
        return base
    audio = load_audio(record["path"], sample_rate)
    return np.concatenate((base, transient_features(audio, sample_rate)))


def audio_features(audio: np.ndarray, sample_rate: int, seconds: float, feature_set: str) -> np.ndarray:
    base = feature_from_audio(audio, sample_rate, seconds)
    if feature_set == "baseline":
        return base
    return np.concatenate((base, transient_features(audio, sample_rate)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="HL-CEAD root containing Manufacturer/Model/RPM/*.wav")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--calibration-folds", type=int, default=4)
    parser.add_argument("--feature-set", choices=("baseline", "transient"), default="baseline")
    parser.add_argument("--model-out", type=Path, default=Path("ml/saved_models/acoustic_hlcead_novelty.joblib"))
    parser.add_argument("--report-out", type=Path, default=Path("ml/evaluation/acoustic_hlcead_novelty.json"))
    args = parser.parse_args()

    records = discover_recordings(args.dataset.expanduser().resolve())
    labels = np.asarray([record["label"] for record in records])
    groups = np.asarray([record["car_id"] for record in records])
    if args.folds < 2 or args.folds > len(np.unique(groups)):
        raise ValueError("--folds must be between 2 and the number of unique cars")

    # All split decisions are finalized before any scaler, detector, or cutoff is fit.
    outer = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
    outer_folds = list(outer.split(np.zeros(len(labels)), labels, groups))
    for fold_no, (tr, te) in enumerate(outer_folds, 1):
        if set(groups[tr]) & set(groups[te]):
            raise ValueError(f"Outer fold {fold_no} has car leakage")
        for subset_name, indices in (("train", tr), ("test", te)):
            missing = set(RPM_CLASSES) - set(labels[indices])
            if missing:
                raise ValueError(f"Outer fold {fold_no} {subset_name} lacks RPM labels: {sorted(missing)}")

    # Feature extraction is deterministic and learns no population parameters.
    # All fitted scaling and detector steps below remain fold-local.
    feature_config = TRANSIENT_FEATURE_CONFIG if args.feature_set == "transient" else FEATURE_CONFIG
    X = np.vstack([recording_features(r, FEATURE_CONFIG["sample_rate"], FEATURE_CONFIG["segment_seconds"], args.feature_set) for r in records])
    oof_scores = np.full(len(records), np.nan)
    oof_thresholds = np.full(len(records), np.nan)
    oof_rpm = np.empty_like(labels)
    synthetic_scores: dict[str, list[float]] = {
        name: [] for name in (
            "synthetic_impulsive_knocks",
            "synthetic_periodic_clicks",
            "synthetic_amplitude_modulation",
        )
    }
    synthetic_thresholds: dict[str, list[float]] = {name: [] for name in synthetic_scores}
    fold_reports = []

    for fold_no, (train_idx, test_idx) in enumerate(outer_folds, 1):
        thresholds, calibration_scores = crossfit_thresholds(
            X[train_idx], labels[train_idx], groups[train_idx], args.calibration_folds,
            seed=1000 * fold_no,
        )
        detectors, fit_counts = fit_rpm_detectors(X[train_idx], labels[train_idx], seed=2000 * fold_no)
        rpm_model = make_estimator().fit(X[train_idx], labels[train_idx])
        routed_rpm = rpm_model.predict(X[test_idx])
        oof_rpm[test_idx] = routed_rpm
        fold_score_by_rpm = {rpm: [] for rpm in RPM_CLASSES}
        for local_index, record_index in enumerate(test_idx):
            record = records[record_index]
            routed = str(routed_rpm[local_index])
            score = float(novelty_score(detectors[routed], X[record_index : record_index + 1])[0])
            threshold = thresholds[routed]
            oof_scores[record_index] = score
            oof_thresholds[record_index] = threshold
            fold_score_by_rpm[routed].append(score)
            audio = load_audio(record["path"], FEATURE_CONFIG["sample_rate"])
            for perturb_index, perturb_name in enumerate(synthetic_scores):
                altered = perturb_audio(audio, FEATURE_CONFIG["sample_rate"], perturb_name)
                altered_feature = audio_features(altered, FEATURE_CONFIG["sample_rate"], FEATURE_CONFIG["segment_seconds"], args.feature_set)
                altered_score = float(novelty_score(detectors[routed], altered_feature.reshape(1, -1))[0])
                synthetic_scores[perturb_name].append(altered_score)
                synthetic_thresholds[perturb_name].append(threshold)

        fold_reports.append({
            "fold": fold_no,
            "train_cars": int(len(np.unique(groups[train_idx]))),
            "test_cars": int(len(np.unique(groups[test_idx]))),
            "detector_fit_recordings_by_rpm": fit_counts,
            "threshold_by_rpm": thresholds,
            "calibration_reference_alert_rate_by_rpm": {
                rpm: float(np.mean(np.asarray(calibration_scores[rpm]) > thresholds[rpm]))
                for rpm in RPM_CLASSES
            },
            "heldout_reference_score_distribution_by_routed_rpm": {
                rpm: score_distribution(np.asarray(fold_score_by_rpm[rpm]), thresholds[rpm])
                if fold_score_by_rpm[rpm] else {"count": 0}
                for rpm in RPM_CLASSES
            },
        })

    if np.isnan(oof_scores).any() or np.isnan(oof_thresholds).any():
        raise RuntimeError("Some recordings did not receive an out-of-fold novelty score")

    synthetic_reports = {}
    for perturb_name, scores in synthetic_scores.items():
        scores_array = np.asarray(scores)
        threshold_array = np.asarray(synthetic_thresholds[perturb_name])
        synthetic_reports[perturb_name] = {
            "label": "SYNTHETIC PERTURBATIONS; not genuine engine faults",
            "score_distribution": {
                "count": int(len(scores_array)),
                "mean": float(np.mean(scores_array)),
                "std": float(np.std(scores_array)),
                "median": float(np.median(scores_array)),
                "p95": float(np.quantile(scores_array, 0.95)),
            },
            "alert_rate_against_vehicle_grouped_thresholds": float(np.mean(scores_array > threshold_array)),
        }

    # OOF cross-fitted scores calibrate the deployment thresholds. The final
    # model and each scaler are then refit on all reference recordings.
    final_thresholds, final_calibration = crossfit_thresholds(
        X, labels, groups, args.calibration_folds, seed=9000,
    )
    final_detectors, final_fit_counts = fit_rpm_detectors(X, labels, seed=9100)
    bundle = {
        "model_version": "hlcead-rpm-novelty-v1",
        "detectors_by_rpm": final_detectors,
        "thresholds_by_rpm": final_thresholds,
        "expected_reference_alert_rate": EXPECTED_REFERENCE_ALERT_RATE,
        "calibration_method": "95th percentile of scores from GroupKFold cross-fitted training-car reference sets, per RPM",
        "feature_config": feature_config,
        "reference_dataset": "HL-CEAD; unlabeled acoustically, not verified healthy",
        "fit_recordings_by_rpm": final_fit_counts,
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)

    report = {
        "dataset": "HL-CEAD",
        "feature_set": args.feature_set,
        "task": "RPM-conditioned acoustic novelty relative to an unlabeled reference set; not fault detection",
        "recordings": len(records),
        "cars": int(len(np.unique(groups))),
        "rpm_recording_counts": {rpm: int(np.sum(labels == rpm)) for rpm in RPM_CLASSES},
        "outer_evaluation": "5-fold StratifiedGroupKFold by manufacturer/model/recording stem; all preprocessing estimators and thresholds fit inside training-car fold",
        "threshold_calibration": {
            "method": "4-fold GroupKFold cross-fitted score distribution on outer-training cars, with separate threshold by RPM",
            "nominal_expected_reference_alert_rate": EXPECTED_REFERENCE_ALERT_RATE,
            "final_crossfit_training_reference_alert_rate_by_rpm": {
                rpm: float(np.mean(np.asarray(final_calibration[rpm]) > final_thresholds[rpm]))
                for rpm in RPM_CLASSES
            },
            "limitation": "Nominal rate is relative to cross-fitted HL-CEAD reference scores; it is not a calibrated false-fault rate and need not hold for future vehicles or conditions.",
        },
        "outer_folds": fold_reports,
        "oof_rpm_routing_accuracy": float(accuracy_score(labels, oof_rpm)),
        "oof_reference_score_distribution_by_routed_rpm": {
            rpm: {
                "count": int(np.sum(oof_rpm == rpm)),
                "mean": float(np.mean(oof_scores[oof_rpm == rpm])),
                "std": float(np.std(oof_scores[oof_rpm == rpm])),
                "median": float(np.median(oof_scores[oof_rpm == rpm])),
                "p95": float(np.quantile(oof_scores[oof_rpm == rpm], 0.95)),
                "alert_rate": float(np.mean(oof_scores[oof_rpm == rpm] > oof_thresholds[oof_rpm == rpm])),
            }
            if np.any(oof_rpm == rpm) else {"count": 0}
            for rpm in RPM_CLASSES
        },
        "oof_reference_alert_rate": float(np.mean(oof_scores > oof_thresholds)),
        "synthetic_perturbation_evaluation": synthetic_reports,
        "generalization_statement": "Scores and alert rates are evaluated on cars absent from the detector/scaler fit. Because HL-CEAD has no verified normal/fault status, this only measures reference-distribution transfer to unseen cars; it cannot establish fault sensitivity, specificity, or diagnosis.",
        "model_artifact": str(args.model_out),
    }
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "oof_reference_alert_rate": report["oof_reference_alert_rate"],
        "oof_score_distribution_by_routed_rpm": report["oof_reference_score_distribution_by_routed_rpm"],
        "synthetic_perturbations": report["synthetic_perturbation_evaluation"],
        "model": str(args.model_out),
        "report": str(args.report_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
