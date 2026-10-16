"""Build a read-only dashboard export from selected upstream carOBD CSVs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_URL = "https://github.com/eron93br/carOBD"
SOURCE_COMMIT_REF = "master"
ROW_LIMIT_PER_FILE = 120
FIELD_MAP = {
    "engine_rpm": ("ENGINE_RPM", "rpm"),
    "engine_load_percent": ("ENGINE_LOAD", "%"),
    "map_kpa": ("INTAKE_MANIFOLD_PRESSURE", "kPa"),
    "throttle_percent": ("THROTTLE", "%"),
    "coolant_temperature_degC": ("COOLANT_TEMPERATURE", "degC"),
    "short_term_fuel_trim_percent": ("SHORT_TERM_FUEL_TRIM_BANK_1", "%"),
    "long_term_fuel_trim_percent": ("LONG_TERM_FUEL_TRIM_BANK_1", "%"),
    "ignition_timing_advance_degBTDC": ("TIMING_ADVANCE", "degBTDC"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        raw_header = next(reader)
        # The upstream header appends an empty unit marker, e.g. ``ENGINE_RPM ()``.
        # Keep the semantic source names while retaining the original CSV files.
        header = [item.strip().split(" (")[0] for item in raw_header]
        while header and not header[-1]:
            header.pop()
        rows: list[dict[str, Any]] = []
        for raw in reader:
            values = list(raw[: len(header)])
            values += [""] * (len(header) - len(values))
            rows.append(dict(zip(header, (value.strip() for value in values))))
    return header, rows


def number(value: str) -> float | None:
    return None if value == "" else float(value)


def build_export(root: str | Path, *, row_limit: int = ROW_LIMIT_PER_FILE) -> dict[str, Any]:
    root = Path(root)
    raw_root = root / "raw"
    source_files = [raw_root / "idle1.csv", raw_root / "drive1.csv"]
    if any(not path.is_file() for path in source_files):
        raise FileNotFoundError("expected imported carOBD raw CSVs are missing")

    trajectories: list[dict[str, Any]] = []
    total_rows = 0
    header_columns: list[str] | None = None
    for path in source_files:
        header, rows = read_rows(path)
        if header_columns is None:
            header_columns = header
        elif header != header_columns:
            raise ValueError("selected carOBD files do not have identical columns")
        selected = rows[:row_limit]
        records: list[dict[str, Any]] = []
        for index, row in enumerate(selected):
            record: dict[str, Any] = {
                "record_index": index,
                "timestamp_s": number(row.get("ENGINE_RUN_TINE", "")),
                "source_values": {column: number(row.get(column, "")) for column in header},
            }
            record["mapped"] = {
                name: number(row.get(source, "")) for name, (source, _unit) in FIELD_MAP.items()
            }
            records.append(record)
        total_rows += len(records)
        timestamps = [item["timestamp_s"] for item in records if item["timestamp_s"] is not None]
        deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
        trajectories.append({
            "trajectory_id": path.stem,
            "mode": "idle" if path.stem.startswith("idle") else "drive",
            "source_file": path.name,
            "source_sha256": sha256(path),
            "source_row_count": len(rows),
            "exported_row_count": len(records),
            "observed_timestamp_s": {"min": min(timestamps), "max": max(timestamps)},
            "observed_timestamp_deltas_s": sorted(set(deltas)),
            "records": records,
        })

    manifest = {
        "format_version": "engineguard-real-obd-carobd/1",
        "data_kind": "real_recorded_ecu_telemetry",
        "diagnostic_label": "unlabeled",
        "source": {
            "repository": SOURCE_URL,
            "ref": SOURCE_COMMIT_REF,
            "vehicle": "Toyota Etios 2014, 1496 CC (per upstream README)",
            "acquisition": "Carloop OBD-II logger; upstream README states nominal 1 Hz",
            "usage_terms": "Upstream README describes repository as open-source/free and requests citation; no formal LICENSE file was found.",
        },
        "selected_files": [item["source_file"] for item in trajectories],
        "source_columns": header_columns,
        "timestamp": {
            "source_column": "ENGINE_RUN_TINE",
            "source_unit": "seconds since engine start (inferred from field name and values; source header leaves units blank)",
            "observed_rate_note": "Rows can repeat timestamps; no resampling or interpolation was applied.",
        },
        "field_mapping": {
            name: {"source_column": source, "export_unit": unit, "source_header_unit": "unspecified"}
            for name, (source, unit) in FIELD_MAP.items()
        },
        "missingness_policy": "Blank source cells remain null; no values were imputed or fabricated.",
        "trajectories": [{key: value for key, value in item.items() if key != "records"} for item in trajectories],
        "exported_row_count": total_rows,
        "limitations": [
            "Real recorded telemetry, but no fault labels or diagnosis.",
            "Only idle1 and drive1 were imported; this is not a vehicle-population sample.",
            "The source README states 1 Hz, while duplicate timestamps and observed deltas are preserved.",
            "Source headers do not declare units; mapped units follow standard OBD PID conventions and are marked as inferred.",
            "No ignition/fault diagnosis should be inferred from these records.",
        ],
    }
    dashboard = {
        "format_version": "engineguard-dashboard-real-obd/1",
        "data_kind": "real_recorded_ecu_telemetry",
        "diagnostic_label": "unlabeled",
        "provenance": {"manifest": "manifest.json", "source_repository": SOURCE_URL},
        "units": {name: unit for name, (_source, unit) in FIELD_MAP.items()},
        "sampling": {"nominal_hz": 1.0, "timestamp_field": "timestamp_s", "resampled": False},
        "trajectories": trajectories,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "dashboard_timeseries.json").write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    build_export(args.root)
