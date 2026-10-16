"""Validation for the shared EngineGuard three-sensor prediction envelope."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any


SENSOR_TYPES = {"acoustic", "vibration", "ckp"}
STATUSES = {"available", "unavailable", "error"}
QUALITY_STATUSES = {"good", "poor", "not_assessed", "unavailable"}
REQUIRED_FIELDS = {
    "schema_version", "prediction_id", "event_id", "sensor_type", "status",
    "engine_id", "session_id", "measurement_interval", "rpm", "anomaly",
    "fault_prediction", "fault_probabilities", "signal_quality",
    "model_version", "limitations", "error", "native_output",
}


class SchemaValidationError(ValueError):
    """Raised when a prediction does not conform to the shared contract."""


def _fail(message: str) -> None:
    raise SchemaValidationError(message)


def _nonempty_string(value: Any, field: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be a non-empty string" + (" or null" if nullable else ""))


def _number(value: Any, field: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{field} must be a finite number" + (" or null" if nullable else ""))


def _timestamp(value: Any, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        _fail(f"{field} must be an ISO-8601 date-time or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError(f"{field} must be an ISO-8601 date-time or null") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{field} must include a timezone")
    return parsed


def validate_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    """Validate and return a prediction mapping, raising on malformed input.

    The returned value is the input mapping; validation does not fill defaults or
    manufacture sensor data. All envelope keys are required, with unsupported
    values represented explicitly as null.
    """
    if not isinstance(prediction, dict):
        _fail("prediction must be an object")
    missing = REQUIRED_FIELDS - prediction.keys()
    extra = prediction.keys() - REQUIRED_FIELDS
    if missing:
        _fail(f"missing required fields: {', '.join(sorted(missing))}")
    if extra:
        allowed = {"provenance"}
        unexpected = extra - allowed
        if unexpected:
            _fail(f"unexpected fields: {', '.join(sorted(unexpected))}")

    if prediction["schema_version"] != "1.0":
        _fail("schema_version must be '1.0'")
    _nonempty_string(prediction["prediction_id"], "prediction_id")
    for field in ("event_id", "engine_id", "session_id"):
        _nonempty_string(prediction[field], field, nullable=True)
    sensor_type = prediction["sensor_type"]
    if not isinstance(sensor_type, str) or sensor_type not in SENSOR_TYPES:
        _fail("sensor_type must be acoustic, vibration, or ckp")
    status = prediction["status"]
    if not isinstance(status, str) or status not in STATUSES:
        _fail("status must be available, unavailable, or error")

    interval = prediction["measurement_interval"]
    if not isinstance(interval, dict) or set(interval) != {"start_utc", "end_utc"}:
        _fail("measurement_interval must contain exactly start_utc and end_utc")
    start = _timestamp(interval["start_utc"], "measurement_interval.start_utc")
    end = _timestamp(interval["end_utc"], "measurement_interval.end_utc")
    if (start is None) != (end is None):
        _fail("measurement interval must provide both endpoints or neither")
    if start is not None and end is not None and end <= start:
        _fail("measurement_interval.end_utc must be after start_utc")

    rpm = prediction["rpm"]
    if rpm is not None:
        if not isinstance(rpm, dict) or set(rpm) != {"value", "confidence", "source"}:
            _fail("rpm must contain value, confidence, and source")
        _number(rpm["value"], "rpm.value")
        if rpm["value"] <= 0:
            _fail("rpm.value must be positive")
        confidence = rpm["confidence"]
        _number(confidence, "rpm.confidence", nullable=True)
        if confidence is not None and not 0 <= confidence <= 1:
            _fail("rpm.confidence must be between 0 and 1")
        _nonempty_string(rpm["source"], "rpm.source")

    anomaly = prediction["anomaly"]
    if anomaly is not None:
        if not isinstance(anomaly, dict) or set(anomaly) != {"score", "threshold", "is_unusual", "score_name"}:
            _fail("anomaly must contain score, threshold, is_unusual, and score_name")
        _number(anomaly["score"], "anomaly.score")
        _number(anomaly["threshold"], "anomaly.threshold")
        if not isinstance(anomaly["is_unusual"], bool):
            _fail("anomaly.is_unusual must be boolean")
        _nonempty_string(anomaly["score_name"], "anomaly.score_name")

    fault_prediction = prediction["fault_prediction"]
    _nonempty_string(fault_prediction, "fault_prediction", nullable=True)
    probabilities = prediction["fault_probabilities"]
    if probabilities is not None:
        if not isinstance(probabilities, dict) or not probabilities:
            _fail("fault_probabilities must be a non-empty object or null")
        for label, probability in probabilities.items():
            _nonempty_string(label, "fault_probabilities class label")
            _number(probability, f"fault_probabilities.{label}")
            if not 0 <= probability <= 1:
                _fail(f"fault_probabilities.{label} must be between 0 and 1")

    quality = prediction["signal_quality"]
    if not isinstance(quality, dict) or set(quality) != {"status", "metrics"}:
        _fail("signal_quality must contain status and metrics")
    if not isinstance(quality["status"], str) or quality["status"] not in QUALITY_STATUSES:
        _fail("signal_quality.status must be good, poor, not_assessed, or unavailable")
    if not isinstance(quality["metrics"], dict):
        _fail("signal_quality.metrics must be an object")
    _nonempty_string(prediction["model_version"], "model_version", nullable=True)
    if not isinstance(prediction["limitations"], list) or any(not isinstance(item, str) for item in prediction["limitations"]):
        _fail("limitations must be an array of strings")
    provenance = prediction.get("provenance", {"kind": "unknown", "source_id": None})
    if not isinstance(provenance, dict) or set(provenance) != {"kind", "source_id"}:
        _fail("provenance must contain kind and source_id")
    if provenance["kind"] not in {"acquisition", "dataset", "synthetic", "unknown"}:
        _fail("provenance.kind must be acquisition, dataset, synthetic, or unknown")
    _nonempty_string(provenance["source_id"], "provenance.source_id", nullable=True)
    if provenance["kind"] == "acquisition" and provenance["source_id"] is None:
        _fail("acquisition provenance requires a source_id")
    if provenance["kind"] == "unknown" and provenance["source_id"] is not None:
        _fail("unknown provenance must have a null source_id")
    error = prediction["error"]
    if error is not None:
        if not isinstance(error, dict) or set(error) != {"code", "message"}:
            _fail("error must contain code and message or be null")
        _nonempty_string(error["code"], "error.code")
        _nonempty_string(error["message"], "error.message")
    native_output = prediction["native_output"]
    if native_output is not None and not isinstance(native_output, dict):
        _fail("native_output must be an object or null")
    if status == "error" and error is None:
        _fail("error status requires an error object")
    if status != "error" and error is not None:
        _fail("error object is only permitted when status is error")
    if status != "available" and (rpm is not None or anomaly is not None or probabilities is not None or fault_prediction is not None):
        _fail("unavailable/error predictions must not contain sensor predictions")
    return prediction


def unavailable_prediction(sensor_type: str, *, reason: str, prediction_id: str, event_id: str | None = None) -> dict[str, Any]:
    """Construct an explicit unavailable envelope without sensor measurements."""
    result = {
        "schema_version": "1.0", "prediction_id": prediction_id, "event_id": event_id,
        "sensor_type": sensor_type, "status": "unavailable", "engine_id": None,
        "session_id": None, "measurement_interval": {"start_utc": None, "end_utc": None},
        "rpm": None, "anomaly": None, "fault_prediction": None,
        "fault_probabilities": None,
        "signal_quality": {"status": "unavailable", "metrics": {}},
        "provenance": {"kind": "unknown", "source_id": None},
        "model_version": None, "limitations": [reason], "error": None, "native_output": None,
    }
    return validate_prediction(result)
