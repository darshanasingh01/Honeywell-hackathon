"""Authoritative ML-owned V2 contract, separate from the retained draft and V1."""
import json
from pathlib import Path

import numpy as np

from .ecu_v2 import finite_number
from .schema_136 import FEATURE_COLUMNS as V1_COLUMNS

MANIFEST = json.loads(Path(__file__).with_name("schema_v2.json").read_text(encoding="utf-8"))
FEATURE_COLUMNS = tuple(MANIFEST["feature_columns"])
assert FEATURE_COLUMNS[:136] == V1_COLUMNS
SCHEMA_VERSION = MANIFEST["schema_version"]


def validate_row(row):
    """Validate a complete feature row; metadata never enters the predictor vector."""
    expected = set(FEATURE_COLUMNS) | set(MANIFEST["metadata_columns"])
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError("exact V2 feature and metadata keys required")
    for name in MANIFEST["provenance_required"] + ["run_id"]:
        if not isinstance(row[name], str) or not row[name].strip():
            raise ValueError(f"nonempty provenance/identifier required: {name}")
    if row["data_kind"] != "synthetic_simulated_engine_conditions":
        raise ValueError("only approved synthetic scope is supported")
    if row["label"] not in MANIFEST["class_labels"]:
        raise ValueError("invalid condition label")
    if type(row["window_id"]) is not int or row["window_id"] < 0:
        raise ValueError("window_id must be a nonnegative integer")
    if finite_number(row["t_start"]) < 0 or finite_number(row["t_end"]) <= row["t_start"]:
        raise ValueError("invalid window")
    if finite_number(row["timestamp_resolution_s"]) <= 0:
        raise ValueError("positive timestamp resolution required")
    for group in MANIFEST["sensor_groups"]:
        if type(row["available_" + group["name"]]) is not bool:
            raise ValueError("availability must be boolean")
    for source in ("ckp", "map", "engine_load"):
        key = source + "_signal_quality_status"
        if row[key] not in MANIFEST["metadata_semantics"][key]:
            raise ValueError("invalid quality status")
        if (row[key] == "unavailable") != (not row["available_" + source]):
            raise ValueError("inconsistent availability and quality")
    for name, spec in MANIFEST["feature_specs"].items():
        permitted = all(row["available_" + g] for g in spec["requires_available_groups"])
        for rule in spec["requires_quality"]:
            if rule != "available_source=true":
                key, required = rule.split("=")
                permitted = permitted and row[key] == required
        value = row[name]
        missing = value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
        if not permitted:
            if not missing:
                raise ValueError(f"unavailable/invalid feature must be null: {name}")
            continue
        if missing and spec.get("nullable_when_unobservable") and row["ckp_signal_quality_status"] == "poor_quality":
            continue
        if missing:
            raise ValueError(f"available feature is missing: {name}")
        number = finite_number(value)
        low, high = spec["expected_range"]
        if (low is not None and number < low) or (high is not None and number > high):
            raise ValueError(f"out-of-range feature: {name}")
        if spec["dtype"] == "integer" and number != int(number):
            raise ValueError(f"integer count required: {name}")
    return {name: row[name] for name in FEATURE_COLUMNS}


def validate_frame(frame):
    """Validate a training handoff, without fitting or generating data."""
    import pandas as pd
    if frame.empty or list(frame.columns) != MANIFEST["metadata_columns"] + list(FEATURE_COLUMNS):
        raise ValueError("nonempty table in exact manifest order required")
    for row in frame.to_dict(orient="records"):
        validate_row(row)
    if frame.duplicated(["run_id", "window_id"]).any():
        raise ValueError("duplicate run/window")
    if (frame.groupby("run_id")["label"].nunique() != 1).any():
        raise ValueError("one condition per run required")
    if set(frame["label"]) != set(MANIFEST["class_labels"]):
        raise ValueError("all five classes required for evaluation")
    for _, run in frame.groupby("run_id"):
        for key in MANIFEST["provenance_required"]:
            if run[key].nunique() != 1:
                raise ValueError("provenance must be consistent within each run")
    return frame.loc[:, FEATURE_COLUMNS].copy(), frame["label"].copy(), frame["run_id"].copy()
