"""Predict a NUST-style healthy/faulty bearing condition from one CSV recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

try:
    from .bearing_condition import CHANNEL, FEATURE_NAMES, MODEL_VERSION, extract_time_domain_features, load_recording
    from .limitations import CLASS_SCOPE, UNCALIBRATED, canonicalize_limitations
except ImportError:  # Support `python ml/vibration/predict_bearing_condition.py`.
    from bearing_condition import CHANNEL, FEATURE_NAMES, MODEL_VERSION, extract_time_domain_features, load_recording
    from limitations import CLASS_SCOPE, UNCALIBRATED, canonicalize_limitations


def predict_recording(signal_path: str | Path, model_path: str | Path) -> dict:
    bundle = joblib.load(model_path)
    if bundle.get("model_version") != MODEL_VERSION:
        raise ValueError(f"Unsupported vibration model version: {bundle.get('model_version')!r}")
    if bundle.get("input_channel") != CHANNEL or bundle.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model feature configuration does not match this prediction implementation")
    recording = load_recording(signal_path, require_dataset_label=False)
    features = pd.DataFrame([extract_time_domain_features(recording.samples)], columns=FEATURE_NAMES)
    model = bundle["model"]
    predicted = str(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    class_probabilities = {str(label): float(prob) for label, prob in zip(model.classes_, probabilities)}
    quality = "good" if recording.timestamp_quality["timestamp_uniform"] else "poor"
    limitations = canonicalize_limitations([
        *bundle.get("limitations", []),
        UNCALIBRATED,
        "No sample rate, duration-based window, filtering, or frequency-domain feature was used.",
        CLASS_SCOPE,
    ])
    return {
        "predicted_bearing_condition": predicted,
        "class_probabilities_uncalibrated": class_probabilities,
        "probabilities_calibrated": False,
        "model_version": bundle["model_version"],
        "input_channel": CHANNEL,
        "input_unit": bundle.get("input_unit"),
        "signal_quality": {"status": quality, "metrics": {
            **recording.timestamp_quality,
            "initial_duplicate_removed": recording.removed_initial_duplicate,
            "sample_count_used": int(recording.samples.size),
        }},
        "limitations": limitations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signal", type=Path, help="One NUST-compatible CSV recording")
    parser.add_argument("--model", type=Path,
                        default=Path("ml/saved_models/vibration_nust_bearing_condition.joblib"))
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    result = predict_recording(args.signal, args.model)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
