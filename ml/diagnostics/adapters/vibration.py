"""Adapt NUST bearing-condition inference to the shared diagnostics envelope."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Any

from ..schema import validate_prediction
from ...vibration.limitations import CLASS_SCOPE, UNCALIBRATED, canonicalize_limitations


def _interval(start_utc: str | None, end_utc: str | None) -> dict[str, str | None]:
    if (start_utc is None) != (end_utc is None):
        raise ValueError("provide both --start-utc and --end-utc, or neither")
    if start_utc is None:
        return {"start_utc": None, "end_utc": None}
    parsed = []
    for field, value in (("start_utc", start_utc), ("end_utc", end_utc)):
        try:
            item = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} must be ISO-8601 with a timezone") from exc
        if item.tzinfo is None or item.utcoffset() is None:
            raise ValueError(f"{field} must include a timezone")
        parsed.append(item)
    if parsed[1] <= parsed[0]:
        raise ValueError("--end-utc must be after --start-utc")
    return {"start_utc": start_utc, "end_utc": end_utc}


def wrap_vibration_output(
    native: dict[str, Any], *, prediction_id: str | None = None,
    event_id: str | None = None, engine_id: str | None = None,
    session_id: str | None = None, start_utc: str | None = None,
    end_utc: str | None = None, provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map native model output to the shared schema without inventing provenance."""
    interval = _interval(start_utc, end_utc)
    labels = {"healthy_bearing", "faulty_bearing"}
    predicted = native.get("predicted_bearing_condition")
    probabilities = native.get("class_probabilities_uncalibrated")
    if predicted not in labels:
        raise ValueError("native output must predict a verified NUST bearing-condition class")
    if not isinstance(probabilities, dict) or set(probabilities) != labels:
        raise ValueError("native output probabilities must contain exactly the verified NUST classes")
    quality = native.get("signal_quality") or {"status": "not_assessed", "metrics": {}}
    limitations = canonicalize_limitations([
        *native.get("limitations", []), UNCALIBRATED, CLASS_SCOPE,
    ])
    if event_id is None or engine_id is None or session_id is None or interval["start_utc"] is None:
        limitations.append("Acquisition linkage metadata was not supplied; this result cannot establish cross-sensor synchronization.")
    result = {
        "schema_version": "1.0",
        "prediction_id": prediction_id or str(uuid.uuid4()),
        "event_id": event_id,
        "sensor_type": "vibration",
        "status": "available",
        "engine_id": engine_id,
        "session_id": session_id,
        "measurement_interval": interval,
        "rpm": None,
        "anomaly": None,
        "fault_prediction": predicted,
        "fault_probabilities": probabilities,
        "signal_quality": quality,
        "provenance": provenance or {"kind": "unknown", "source_id": None},
        "model_version": native.get("model_version"),
        "limitations": limitations,
        "error": None,
        "native_output": native,
    }
    return validate_prediction(result)


def _error_envelope(message: str, *, prediction_id: str, event_id: str | None,
                    engine_id: str | None, session_id: str | None,
                    interval: dict[str, str | None]) -> dict[str, Any]:
    return validate_prediction({
        "schema_version": "1.0", "prediction_id": prediction_id, "event_id": event_id,
        "sensor_type": "vibration", "status": "error", "engine_id": engine_id,
        "session_id": session_id, "measurement_interval": interval,
        "rpm": None, "anomaly": None, "fault_prediction": None,
        "fault_probabilities": None,
        "signal_quality": {"status": "unavailable", "metrics": {}},
        "provenance": {"kind": "unknown", "source_id": None},
        "model_version": None,
        "limitations": ["Vibration bearing-condition inference did not produce a prediction."],
        "error": {"code": "vibration_inference_failed", "message": message},
        "native_output": None,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signal", type=Path)
    parser.add_argument("--model", type=Path,
                        default=Path("ml/saved_models/vibration_nust_bearing_condition.joblib"))
    parser.add_argument("--event-id", help="Acquisition-provided event ID")
    parser.add_argument("--engine-id", help="Acquisition-provided engine ID")
    parser.add_argument("--session-id", help="Acquisition-provided session ID")
    parser.add_argument("--start-utc", help="Acquisition-provided timezone-aware start time")
    parser.add_argument("--end-utc", help="Acquisition-provided timezone-aware end time")
    parser.add_argument("--provenance-kind", choices=("acquisition", "dataset", "synthetic", "unknown"), default="unknown")
    parser.add_argument("--provenance-source-id", help="Acquisition or source dataset ID; required for acquisition provenance")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    interval = _interval(args.start_utc, args.end_utc)
    prediction_id = str(uuid.uuid4())
    model_path = args.model if args.model.is_absolute() else root / args.model
    command = [sys.executable, str(root / "ml/vibration/predict_bearing_condition.py"),
               str(args.signal.resolve()), "--model", str(model_path.resolve())]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        try:
            native = json.loads(completed.stdout)
            envelope = wrap_vibration_output(
                native, prediction_id=prediction_id, event_id=args.event_id,
                engine_id=args.engine_id, session_id=args.session_id,
                start_utc=args.start_utc, end_utc=args.end_utc,
                provenance={"kind": args.provenance_kind,
                            "source_id": args.provenance_source_id if args.provenance_kind != "unknown" else None},
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            envelope = _error_envelope(str(exc), prediction_id=prediction_id,
                                       event_id=args.event_id, engine_id=args.engine_id,
                                       session_id=args.session_id, interval=interval)
    else:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        envelope = _error_envelope(detail, prediction_id=prediction_id,
                                   event_id=args.event_id, engine_id=args.engine_id,
                                   session_id=args.session_id, interval=interval)
    encoded = json.dumps(envelope, indent=2)
    if args.output:
        output_path = args.output if args.output.is_absolute() else root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0 if envelope["status"] == "available" else 1


if __name__ == "__main__":
    raise SystemExit(main())
