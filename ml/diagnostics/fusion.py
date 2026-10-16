"""Conservative late fusion for acoustic, vibration, and CKP predictions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from .schema import SchemaValidationError, validate_prediction

SENSOR_TYPES = ("acoustic", "vibration", "ckp")


def _interval(item: dict[str, Any]) -> tuple[str, str] | None:
    interval = item["measurement_interval"]
    start, end = interval["start_utc"], interval["end_utc"]
    return (start, end) if start is not None and end is not None else None


def _overlaps(intervals: list[tuple[str, str]]) -> bool:
    """Return true only for a non-empty common interval, normalized to UTC."""
    def parse(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)

    return max(parse(start) for start, _ in intervals) < min(parse(end) for _, end in intervals)


def _provenance(item: dict[str, Any]) -> dict[str, Any]:
    return item.get("provenance", {"kind": "unknown", "source_id": None})


def fuse_predictions(predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Validate and summarize sensor envelopes without mixing unlike scores.

    Original envelopes are preserved under ``sensor_results``. Evidence with
    poor quality is excluded. ``not_assessed`` evidence may be displayed for
    review, but only good-quality anomalies can establish a fused event.
    Event fusion additionally requires matching acquisition provenance and
    engine/session/event identity plus overlapping, timezone-aware intervals.
    """
    results = list(predictions)
    if not results:
        return {
            "fusion_status": "not_fused", "event_id": None,
            "fusion_decision": "no_evidence", "fault_prediction": None,
            "sensor_results": [], "anomaly_evidence": [],
            "missing_sensors": list(SENSOR_TYPES),
            "fusion_reasons": ["no_sensor_outputs_provided"],
            "limitations": ["No sensor predictions were supplied."],
        }
    for result in results:
        validate_prediction(result)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_type[result["sensor_type"]].append(result)
    missing_sensors = [sensor for sensor in SENSOR_TYPES if sensor not in by_type]

    # Duplicate results for one modality stay visible but cannot count as
    # independent evidence or be selected arbitrarily.
    unique_results = [group[0] for group in by_type.values() if len(group) == 1]
    duplicate_types = sorted(sensor for sensor, group in by_type.items() if len(group) > 1)

    display_evidence = [
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
            "signal_quality": item["signal_quality"]["status"],
        }
        for item in display_evidence
    ]
    # Unknown/not-assessed quality can remain visible but is not sufficient to
    # permit actual event-level fusion.
    eligible = [item for item in display_evidence if item["signal_quality"]["status"] == "good"]
    fusion_candidates = [
        item for item in unique_results
        if item["status"] == "available" and item["signal_quality"]["status"] == "good"
    ]

    reasons: list[str] = []
    gate_reasons: list[str] = []

    def block(reason: str) -> None:
        reasons.append(reason)
        gate_reasons.append(reason)
    if missing_sensors:
        reasons.append("missing_sensor_outputs:" + ",".join(missing_sensors))
    if duplicate_types:
        reasons.append("duplicate_sensor_outputs:" + ",".join(duplicate_types))
    if any(item["status"] != "available" for item in results):
        reasons.append("unavailable_or_error_sensor_outputs_present")
    if any(item["signal_quality"]["status"] == "poor" for item in results):
        reasons.append("poor_quality_sensor_outputs_present")
    if any(item["signal_quality"]["status"] == "not_assessed" for item in results):
        reasons.append("signal_quality_not_assessed_for_one_or_more_sensors")

    if len(results) == 1:
        fusion_status = "single_sensor"
    elif len(eligible) < 2:
        fusion_status = "not_fused"
        reasons.append("fewer_than_two_good_quality_anomaly_outputs")
        candidate_sources = [item for item in unique_results if item["status"] == "available"]
        candidate_provenances = [_provenance(item) for item in candidate_sources]
        if len(candidate_provenances) > 1 and any(
            p["kind"] != "acquisition" or not p["source_id"]
            for p in candidate_provenances
        ):
            reasons.append("acquisition_provenance_unknown_or_not_acquisition_linked")
        elif len(candidate_provenances) > 1 and any(
            p["source_id"] != candidate_provenances[0]["source_id"]
            for p in candidate_provenances[1:]
        ):
            reasons.append("conflicting_acquisition_provenance")
    else:
        required_metadata = ("engine_id", "session_id", "event_id")
        for field in required_metadata:
            if any(not item[field] for item in fusion_candidates):
                block(f"missing_{field}")
            elif any(item[field] != fusion_candidates[0][field] for item in fusion_candidates[1:]):
                block(f"conflicting_{field}")
        intervals = [_interval(item) for item in fusion_candidates]
        if any(interval is None for interval in intervals):
            block("missing_measurement_timestamps")
        elif not _overlaps([interval for interval in intervals if interval is not None]):
            block("measurement_intervals_do_not_overlap")
        provenances = [_provenance(item) for item in fusion_candidates]
        if any(p["kind"] != "acquisition" or not p["source_id"] for p in provenances):
            block("acquisition_provenance_unknown_or_not_acquisition_linked")
        elif any(p["source_id"] != provenances[0]["source_id"] for p in provenances[1:]):
            block("conflicting_acquisition_provenance")
        fusion_status = "fused" if not gate_reasons else "not_fused"

    alerts = [entry for entry in anomaly_evidence if entry["is_unusual"]]
    good_alerts = {item["sensor_type"] for item in eligible if item["anomaly"]["is_unusual"]}
    if fusion_status == "fused" and len(good_alerts) >= 2:
        decision = "multi_sensor_corroboration"
    elif alerts:
        decision = "review_recommended"
    else:
        decision = "no_anomaly_alert"

    limitations = [
        "Anomaly agreement is not a confirmed fault diagnosis.",
        "Sensor scores, thresholds, RPM estimates, and class scores are kept separate; no cross-model probability is calculated.",
        "RPM is operating context and is not used as fault evidence.",
    ]
    if fusion_status == "not_fused" and len(results) > 1:
        limitations.append("Event-level fusion was withheld because quality, acquisition provenance, metadata, or interval checks did not pass.")
    if duplicate_types:
        limitations.append("Duplicate sensor types were retained but excluded as ambiguous independent evidence: " + ", ".join(duplicate_types) + ".")
    if not anomaly_evidence:
        limitations.append("Unavailable, error, poor-quality, or unsupported anomaly outputs contribute no anomaly evidence.")
    if any(_provenance(item)["kind"] in {"dataset", "synthetic", "unknown"} for item in results):
        limitations.append("Dataset, synthetic, or unknown-provenance outputs cannot establish synchronized acquisition events.")

    return {
        "fusion_status": fusion_status,
        "event_id": eligible[0]["event_id"] if fusion_status == "fused" else None,
        "fusion_decision": decision,
        "fault_prediction": None,
        "sensor_results": results,
        "anomaly_evidence": anomaly_evidence,
        "missing_sensors": missing_sensors,
        "fusion_reasons": reasons,
        "limitations": limitations,
    }


def fuse_sensor_outputs(
    *,
    acoustic: dict[str, Any] | None = None,
    vibration: dict[str, Any] | None = None,
    ckp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unified named interface for any available subset of the three sensors."""
    named = (("acoustic", acoustic), ("vibration", vibration), ("ckp", ckp))
    for expected_type, prediction in named:
        if prediction is not None:
            validate_prediction(prediction)
            if prediction["sensor_type"] != expected_type:
                raise SchemaValidationError(
                    f"{expected_type} input must have sensor_type '{expected_type}'"
                )
    return fuse_predictions([prediction for _, prediction in named if prediction is not None])


__all__ = ["fuse_predictions", "fuse_sensor_outputs", "SchemaValidationError"]
