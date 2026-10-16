"""Read-only compact nine-sensor export from an existing V2 run handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .schema_v2 import FEATURE_COLUMNS, MANIFEST
from .verify_producer_v2 import verify_directory

DEFAULT_DATASET = Path("ml/datasets/synthetic_conditions_v2/training")
DEFAULT_OUTPUT = Path("ml/synthetic_conditions/sensor_exports_v2")
EXPORT_VERSION = "engineguard-sensor-export/1"
REQUESTED_CONDITIONS = ("normal", "misfire", "overheating")

SENSOR_GROUPS = {
    "microphone": {"sample_rate_hz": 25600, "units": "synthetic waveform amplitude",
                   "source": "microphone_audio", "raw_trace": False},
    "knock": {"sample_rate_hz": 25600, "units": "synthetic waveform amplitude",
              "source": "knock_sensor", "raw_trace": False},
    "ckp": {"sample_rate_hz": None, "units": "timestamp_s, tooth index, seconds, rpm, rad/s",
            "source": "timestamped 60-2 tooth edges", "raw_trace": False},
    "engine_rpm": {"sample_rate_hz": 10, "units": "rpm", "source": "PID 0x0C", "raw_trace": False},
    "engine_load": {"sample_rate_hz": 10, "units": "fraction and percent", "source": "PID 0x04", "raw_trace": False},
    "map": {"sample_rate_hz": 10, "units": "kPa absolute", "source": "PID 0x0B", "raw_trace": False},
    "throttle": {"sample_rate_hz": 10, "units": "percent", "source": "PID 0x11", "raw_trace": False},
    "coolant": {"sample_rate_hz": 10, "units": "degC", "source": "PID 0x05", "raw_trace": False},
    "fuel_trims": {"sample_rate_hz": 10, "units": "percent", "source": "PIDs 0x06 and 0x07", "raw_trace": False},
    "ignition_timing": {"sample_rate_hz": 10, "units": "degrees BTDC", "source": "PID 0x0E", "raw_trace": False},
    "accelerometer": {"sample_rate_hz": 25600, "units": "synthetic waveform amplitude",
                       "source": "accel_x, accel_y, accel_z", "raw_trace": False},
}
WAVEFORM_PREFIXES = {"microphone": ("mic_",), "knock": ("knock_",),
                     "accelerometer": ("accel_x_", "accel_y_", "accel_z_")}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_summary(row):
    return {group: {name: float(row[name]) for name in FEATURE_COLUMNS
                    if any(name.startswith(prefix) for prefix in prefixes)}
            for group, prefixes in WAVEFORM_PREFIXES.items()}


def _telemetry_samples(run, start, end):
    source = pd.DataFrame(run["telemetry"])
    selected = source[(source["timestamp_s"] >= start) & (source["timestamp_s"] < end)]
    result = []
    for record in selected.to_dict(orient="records"):
        result.append({"timestamp_s": float(record["timestamp_s"]),
                       "engine_rpm": float(record["engine_rpm"]),
                       "engine_load_fraction": float(record["engine_load"]),
                       "engine_load_pct": float(record["engine_load_pct"]),
                       "map_kpa": float(record["map_kpa"]),
                       "throttle_pct": float(record["throttle_pct"]),
                       "coolant_temp_c": float(record["coolant_temp_c"]),
                       "stft_pct": float(record["stft_pct"]),
                       "ltft_pct": float(record["ltft_pct"]),
                       "ignition_timing_deg_btdc": float(record["ignition_timing"])})
    return result


def export_one(frame, run, manifest, *, source_dataset_sha256):
    run_id = run["run_id"]
    rows = frame[(frame.run_id == run_id) & (frame.window_id == 0)]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one window 0 for {run_id}")
    row = rows.iloc[0]
    start, end = float(row.t_start), float(row.t_end)
    telemetry = _telemetry_samples(run, start, end)
    ckp = [{"timestamp_s": float(t), "tooth_index": int(i)}
           for t, i in run["ckp_events"] if start <= t < end]
    if not telemetry or not ckp:
        raise ValueError(f"missing synchronized evidence for {run_id}/0")
    if any(record["timestamp_s"] < start or record["timestamp_s"] >= end for record in telemetry + ckp):
        raise ValueError("export evidence crossed window boundary")
    feature_summary = _feature_summary(row)
    sensors = {name: {**spec, "available": True,
                      "evidence": ("feature_summary" if name in feature_summary else
                                   "timestamped_events" if name == "ckp" else "telemetry_samples")}
               for name, spec in SENSOR_GROUPS.items()}
    sensors["ckp"]["signal_quality_status"] = str(row.ckp_signal_quality_status)
    sensors["map"]["signal_quality_status"] = str(row.map_signal_quality_status)
    sensors["engine_load"]["signal_quality_status"] = str(row.engine_load_signal_quality_status)
    return {
        "format_version": EXPORT_VERSION,
        "data_kind": "synthetic_simulated_engine_conditions",
        "real_engine_measurements": False,
        "condition_label": str(row.label),
        "run_id": run_id,
        "window_id": int(row.window_id),
        "window": {"t_start": start, "t_end": end, "semantics": "half-open [t_start,t_end)"},
        "sensor_groups": sensors,
        "telemetry_samples": telemetry,
        "ckp_events": ckp,
        "waveform_evidence": {
            "raw_waveform_available": False,
            "raw_waveform_note": "Raw microphone, knock and accelerometer arrays were not retained in the V2 handoff.",
            "feature_summary_available": True,
            "feature_summary": feature_summary,
        },
        "synchronization": {
            "same_generated_run": True, "same_window": True,
            "telemetry_sample_count": len(telemetry), "ckp_event_count": len(ckp),
            "timestamp_clock": "shared run clock seconds",
            "ckp_timestamp_resolution_s": float(row.timestamp_resolution_s),
        },
        "provenance": {
            "source_id": str(row.source_id), "source_commit": str(row.source_commit),
            "processor_revision": str(row.processor_revision),
            "dataset_schema_version": manifest["schema_version"],
            "dataset_features_sha256": source_dataset_sha256,
            "seed": int(run["seed"]), "producer_version": manifest["producer_version"],
        },
        "limitations": ["All values are synthetic simulator outputs.",
            "Raw fast-sensor traces are unavailable; feature summaries are not waveform replacements.",
            "This export is a synchronized data handoff, not a diagnostic prediction.",
            "No real-engine measurement or calibrated physical-unit validation is established."],
    }


def export_samples(dataset=DEFAULT_DATASET, output=DEFAULT_OUTPUT,
                   conditions=REQUESTED_CONDITIONS):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing export directory: {output}")
    conditions = tuple(conditions)
    if set(conditions) != set(REQUESTED_CONDITIONS):
        raise ValueError("export requires exactly normal, misfire, and overheating")
    # Full read-only source verification; no producer or dataset generation.
    verification = verify_directory(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((dataset / "run_evidence.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(dataset / "features.csv", float_precision="round_trip")
    by_condition = {run["condition"]: run for run in evidence if run["seed"] == 3200}
    if set(by_condition) != set(MANIFEST["class_labels"]):
        raise ValueError("expected seed-3200 evidence for every declared class")
    output.mkdir(parents=True, exist_ok=False)
    exported = {}
    for condition in conditions:
        payload = export_one(frame, by_condition[condition], manifest,
                             source_dataset_sha256=manifest["features_sha256"])
        path = output / f"{condition}_window0.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        exported[condition] = {"path": path.name, "sha256": _sha256(path),
                               "run_id": payload["run_id"], "window_id": payload["window_id"],
                               "condition_label": payload["condition_label"]}
    index = {"format_version": EXPORT_VERSION, "data_kind": "synthetic_simulated_engine_conditions",
             "real_engine_measurements": False, "source_dataset": str(dataset.as_posix()),
             "source_dataset_features_sha256": manifest["features_sha256"],
             "source_verification_status": verification["status"], "sensor_group_count": 10,
             "primary_sensor_group_count": 9, "supporting_sensor_group": "accelerometer",
             "exports": exported,
             "missing_evidence": {"raw_microphone_waveform": True, "raw_knock_waveform": True,
                                   "raw_accelerometer_waveform": True},
             "limitations": ["Synthetic-only export.", "Raw fast waveforms were not retained in the source handoff."]}
    (output / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export_samples(args.dataset, args.output), indent=2))


if __name__ == "__main__":
    main()
