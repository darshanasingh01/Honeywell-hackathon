"""Adapt CKP RPM analysis into the shared EngineGuard diagnostics envelope."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import uuid

from ..schema import validate_prediction
from ...ckp.analyze import analyze_csv


def _interval(start_utc: str | None, end_utc: str | None) -> dict[str, str | None]:
    if (start_utc is None) != (end_utc is None):
        raise ValueError("provide both --start-utc and --end-utc, or neither")
    if start_utc is None:
        return {"start_utc": None, "end_utc": None}
    parsed = []
    for name, value in (("start_utc", start_utc), ("end_utc", end_utc)):
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{name} must be timezone-aware ISO-8601") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware ISO-8601")
        parsed.append(timestamp)
    if parsed[1] <= parsed[0]:
        raise ValueError("end time must be after start time")
    return {"start_utc": start_utc, "end_utc": end_utc}


def wrap_ckp_output(
    native: dict,
    *,
    prediction_id: str | None = None,
    event_id: str | None = None,
    engine_id: str | None = None,
    session_id: str | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
) -> dict:
    """Wrap native RPM analysis without creating acquisition provenance."""
    native = copy.deepcopy(native)
    native.pop("ground_truth_comparison", None)
    interval = _interval(start_utc, end_utc)
    available = native.get("status") == "available" and native.get("estimated_rpm") is not None
    result = {
        "schema_version": "1.0",
        "prediction_id": prediction_id or str(uuid.uuid4()),
        "event_id": event_id,
        "sensor_type": "ckp",
        "status": "available" if available else "unavailable",
        "engine_id": engine_id,
        "session_id": session_id,
        "measurement_interval": interval,
        "rpm": ({
            "value": float(native["estimated_rpm"]),
            "confidence": None,
            "source": "ckp_tooth_timing",
        } if available else None),
        "anomaly": None,
        "fault_prediction": None,
        "fault_probabilities": None,
        "signal_quality": {
            "status": native.get("quality_status", "unavailable") if available else "unavailable",
            "metrics": native.get("metrics", {}),
        },
        "model_version": native.get("model_version"),
        "limitations": list(native.get("limitations", [])),
        "error": None,
        "native_output": native,
    }
    if not available and native.get("reason"):
        result["limitations"].append(f"CKP RPM output is unavailable: {native['reason']}.")
    if native.get("synthetic"):
        result["limitations"].append("Input waveform is synthetic and is not a real CKP sensor measurement.")
    return validate_prediction(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signal", type=Path)
    parser.add_argument("--sample-rate-hz", type=int)
    parser.add_argument("--nominal-teeth", type=int, default=60)
    parser.add_argument("--missing-teeth", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    native = analyze_csv(
        args.signal, sample_rate_hz=args.sample_rate_hz,
        nominal_teeth=args.nominal_teeth, missing_teeth=args.missing_teeth,
    )
    envelope = wrap_ckp_output(native)
    encoded = json.dumps(envelope, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0 if envelope["status"] == "available" else 1


if __name__ == "__main__":
    raise SystemExit(main())
