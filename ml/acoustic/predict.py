"""Emit the acoustic inference contract as one JSON object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from features import extract_features
from preprocess import load_audio, split_audio


def vectorize(audio: np.ndarray, sample_rate: int, segment_seconds: float) -> np.ndarray:
    chunks = split_audio(audio, sample_rate, segment_seconds)
    if not chunks:
        raise ValueError(f"Recording must be at least {segment_seconds:g} seconds long")
    return np.mean([extract_features(chunk, sample_rate) for chunk in chunks], axis=0).reshape(1, -1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--rpm-model", type=Path, default=Path("ml/saved_models/acoustic_hlcead_rpm.joblib"))
    parser.add_argument("--novelty-model", type=Path, default=Path("ml/saved_models/acoustic_hlcead_novelty.joblib"))
    parser.add_argument("--output", type=Path, help="Optional JSON output path; default writes JSON to stdout")
    args = parser.parse_args()

    rpm_bundle = joblib.load(args.rpm_model)
    sample_rate = int(rpm_bundle["sample_rate"])
    segment_seconds = float(rpm_bundle["segment_seconds"])
    audio = load_audio(args.audio, sample_rate)
    features = vectorize(audio, sample_rate, segment_seconds)
    rpm_model = rpm_bundle["model"]
    predicted_rpm = str(rpm_model.predict(features)[0])

    rpm_confidence = None
    limitations = [
        "HL-CEAD RPM scores reflect its limited recording conditions and do not establish fault status.",
        "Acoustic novelty means deviation from the unlabeled HL-CEAD reference distribution; it is not a fault diagnosis.",
    ]
    final_estimator = rpm_model[-1] if hasattr(rpm_model, "__getitem__") else rpm_model
    if getattr(final_estimator, "probability", False) and hasattr(rpm_model, "predict_proba"):
        probability = rpm_model.predict_proba(features)[0]
        rpm_confidence = float(np.max(probability))
    else:
        limitations.append("RPM confidence is null because the saved SVM does not provide calibrated probabilities.")

    novelty_score = None
    novelty_threshold = None
    is_unusual = None
    novelty_path = args.novelty_model
    if novelty_path.is_file():
        novelty_bundle = joblib.load(novelty_path)
        detector = novelty_bundle["detectors_by_rpm"][predicted_rpm]
        novelty_score = float(-detector.decision_function(features)[0])
        novelty_threshold = float(novelty_bundle["thresholds_by_rpm"][predicted_rpm])
        is_unusual = novelty_score > novelty_threshold
        model_version = f"{rpm_bundle.get('model_version', 'hlcead-rpm-v1')}+{novelty_bundle['model_version']}"
    else:
        limitations.append("No novelty artifact is available; novelty fields are null.")
        model_version = rpm_bundle.get("model_version", "hlcead-rpm-v1")

    response = {
        "predicted_rpm": int(predicted_rpm),
        "rpm_confidence": rpm_confidence,
        "acoustic_novelty_score": novelty_score,
        "novelty_threshold": novelty_threshold,
        "is_acoustically_unusual": is_unusual,
        "fault_prediction": None,
        "model_version": model_version,
        "limitations": limitations,
    }
    encoded = json.dumps(response, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
