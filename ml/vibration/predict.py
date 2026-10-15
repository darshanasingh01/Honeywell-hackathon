"""JSON inference for a single vibration window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from features import extract_features
from signals import load_signal, preprocess, segment_signal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signal", type=Path)
    parser.add_argument("--sample-rate-hz", required=True, type=float)
    parser.add_argument("--unit", required=True, help="Physical unit as recorded, e.g. m/s^2 or g")
    parser.add_argument("--column", help="CSV/NPZ signal column/key when needed")
    parser.add_argument("--model", type=Path, default=Path("ml/saved_models/vibration_novelty.joblib"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    if float(args.sample_rate_hz) != float(bundle["sample_rate_hz"]):
        raise ValueError(f"Model expects {bundle['sample_rate_hz']} Hz; received {args.sample_rate_hz} Hz")
    if args.unit != bundle["unit"]:
        raise ValueError(f"Model expects unit {bundle['unit']!r}; received {args.unit!r}")
    signal = preprocess(
        load_signal(args.signal, args.sample_rate_hz, args.unit, args.column),
        **bundle.get("preprocessing", {}),
    )
    windows = segment_signal(signal, float(bundle["window_seconds"]), overlap=0.0)
    if len(windows) != 1:
        raise ValueError(f"Provide exactly one {bundle['window_seconds']:g}s inference window; got {len(windows)} windows")
    features = pd.DataFrame([extract_features(windows[0], signal.sample_rate_hz)])[bundle["feature_names"]]
    model = bundle["model"]
    score = float(-model.decision_function(features)[0])
    threshold = float(bundle["threshold"])
    result = {
        "vibration_anomaly_score": score,
        "vibration_anomaly_threshold": threshold,
        "is_vibration_unusual": bool(score > threshold),
        "predicted_fault_class": None,
        "model_version": bundle["model_version"],
        "limitations": [
            "Novelty indicates deviation from an unlabeled reference distribution, not a confirmed fault.",
            f"Signal units were supplied as {signal.unit!r}; model scores are only comparable under matching sensor, mounting, unit, rate, and preprocessing conditions.",
            "No supervised fault classifier has been validated.",
        ],
    }
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
