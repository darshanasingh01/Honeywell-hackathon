"""Conservative late fusion for acoustic, vibration, and CKP predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .schema import SchemaValidationError, validate_prediction


def _interval(item: dict[str, Any]) -> tuple[str, str] | None:
    interval = item["measurement_interval"]
    start, end = interval["start_utc"], interval["end_utc"]
    return (start, end) if start is not None and end is not None else None


def _overlaps(intervals: list[tuple[str, str]]) -> bool:
    """Return true only for a non-empty common UTC interval (lexical ISO UTC)."""
    # Normalize timestamps to aware UTC datetimes to handle differing offsets.
    from datetime import datetime, timezone

    def parse(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

    latest_start = max(parse(start) for start, _ in intervals)
    earliest_end = min(parse(end) for _, end in intervals)
    return latest_start < earliest_end


def fuse_predictions(predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a subset of sensor outputs without blending unlike scores.

    All inputs are returned unchanged under ``sensor_results``. Duplicate
    sensor types remain visible but count as only one modality for corroboration.
    Actual multi-sensor fusion requires complete, matching acquisition metadata
    and a shared overlapping interval among eligible sensor results.
    """
    results = list(predictions)
    if not results:
        return {
            "fusion_status": "not_fused", "fusion_decision": "no_evidence",
            "fault_prediction": None, "sensor_results": [], "anomaly_evidence": [],
            "limitations": ["No sensor predictions were supplied."],
        }
    for result in results:
        validate_prediction(result)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_type[result["sensor_type"]].append(result)

    # One output per physical sensor type. Exclude an ambiguous type if it was
    # supplied more than once rather than arbitrarily selecting one result.
    unique_results: list[dict[str, Any]] = []
    duplicate_types: list[str] = []
    for sensor_type, same_type in by_type.items():
        if len(same_type) == 1:
            unique_results.append(same_type[0])
        else:
            duplicate_types.append(sensor_type)

    eligible = [
        item for item in unique_results
        if item["status"] == "available"
        and item["signal_quality"]["status"] not in {"poor", "unavailable"}
        and item["anomaly"] is not None
    ]
    anomaly_evidence = [
        {
            "sensor_type": item["sensor_type"],
            "score_name": item["anomaly"]["score_name"],
            "score": item["anomaly"]["score"],
            "threshold": item["anomaly"]["threshold"],
            "is_unusual": item["anomaly"]["is_unusual"],
        }
        for item in eligible
    ]
    alerts = [entry for entry in anomaly_evidence if entry["is_unusual"]]
    distinct_alerts = {entry["sensor_type"] for entry in alerts}

    if len(by_type) == 1:
        fusion_status = "single_sensor"
        metadata_valid = False
    elif len(eligible) < 2:
        fusion_status = "not_fused"
        metadata_valid = False
    else:
        required_metadata = ("engine_id", "session_id", "event_id")
        metadata_valid = all(
            all(item[key] for key in required_metadata) and _interval(item) is not None
            for item in eligible
        )
        if metadata_valid:
            first = eligible[0]
            metadata_valid = all(
                all(item[key] == first[key] for key in required_metadata)
                for item in eligible[1:]
            )
        if metadata_valid:
            metadata_valid = _overlaps([_interval(item) for item in eligible if _interval(item) is not None])
        fusion_status = "fused" if metadata_valid else "not_fused"

    if len(distinct_alerts) >= 2 and fusion_status == "fused":
        decision = "multi_sensor_corroboration"
    elif alerts:
        decision = "review_recommended"
    else:
        decision = "no_anomaly_alert"

    limitations = [
        "Anomaly agreement is not a confirmed fault diagnosis.",
        "Scores and thresholds are reported per sensor and are never averaged or compared across models.",
        "RPM is operating context and is not used as fault evidence.",
    ]
    if fusion_status == "not_fused" and len(by_type) > 1:
        limitations.append(
            "Multi-sensor fusion was withheld: eligible results need matching acquisition-provided engine_id, session_id, event_id, and overlapping intervals."
        )
    if duplicate_types:
        limitations.append(
            "Duplicate sensor types were preserved but excluded as ambiguous evidence: " + ", ".join(sorted(duplicate_types)) + "."
        )
    if not eligible:
        limitations.append("Unavailable, error, poor-quality, or unsupported anomaly outputs contribute no evidence.")

    shared_event = eligible[0]["event_id"] if fusion_status == "fused" else None
    return {
        "fusion_status": fusion_status,
        "event_id": shared_event,
        "fusion_decision": decision,
        "fault_prediction": None,
        "sensor_results": results,
        "anomaly_evidence": anomaly_evidence,
        "limitations": limitations,
    }


__all__ = ["fuse_predictions", "SchemaValidationError"]
