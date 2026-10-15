"""Canonicalize repeated NUST prediction limitations without dropping warnings."""

from __future__ import annotations


UNCALIBRATED = "Class probabilities are uncalibrated model scores, not verified fault probabilities."
CLASS_SCOPE = (
    "Prediction is limited to the dataset's healthy_bearing/faulty_bearing bearing-condition labels; "
    "fault mechanism and severity are unspecified, and this is not a combustion-knock diagnosis."
)


def canonicalize_limitations(items) -> list[str]:
    """Collapse known duplicate/overlapping warnings and retain other unique text."""
    result: list[str] = []
    seen: set[str] = set()
    has_calibration = False
    has_class_scope = False
    for item in items or []:
        if not isinstance(item, str) or not item.strip():
            continue
        text = " ".join(item.split())
        lowered = text.casefold()
        if any(term in lowered for term in ("uncalibrat", "not calibrated probabilities", "not calibrated model scores")):
            has_calibration = True
            continue
        if any(term in lowered for term in (
            "bearing-condition labels", "bearing condition labels", "fault mechanism and severity",
            "combustion-knock diagnosis", "not a combustion-knock classifier",
        )):
            has_class_scope = True
            continue
        key = lowered.rstrip(".")
        if key not in seen:
            seen.add(key)
            result.append(text)
    if has_calibration:
        result.append(UNCALIBRATED)
    if has_class_scope:
        result.append(CLASS_SCOPE)
    return result
