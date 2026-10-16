"""Read-only verification of a serialized synthetic V2 producer handoff."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .ckp_v2 import process_edges
from .ecu_v2 import window_features
from .producer_v2 import CONDITIONS, SOURCE_COMMIT, approved_sources, sha256, validate_dataset
from .schema_v2 import FEATURE_COLUMNS, MANIFEST, SCHEMA_VERSION


def verify_directory(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((directory / "run_evidence.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(directory / "features.csv", float_precision="round_trip")
    report = validate_dataset(frame)
    if (manifest["schema_version"] != SCHEMA_VERSION or manifest["source_commit"] != SOURCE_COMMIT
            or manifest["real_engine_measurements"] is not False
            or manifest["features_sha256"] != sha256((directory / "features.csv").read_bytes())
            or manifest["approved_blob_sha256"] != approved_sources()[-1]
            or list(manifest["feature_producer_paths"]) != list(FEATURE_COLUMNS)):
        raise ValueError("dataset provenance/hash mismatch")
    for name, digest in manifest["ml_source_sha256"].items():
        if Path(name).name != name or sha256(Path(__file__).with_name(name).read_bytes()) != digest:
            raise ValueError("ML source hash mismatch")
    if len(evidence) != frame.run_id.nunique() or len({r["run_id"] for r in evidence}) != len(evidence):
        raise ValueError("run evidence mismatch")
    expected_run_pairs = {(c, s) for c in CONDITIONS for s in manifest["configuration"]["seeds"]}
    if {(r["condition"], r["seed"]) for r in evidence} != expected_run_pairs:
        raise ValueError("condition/seed coverage mismatch")
    minimum_resolution = float("inf")
    for run in evidence:
        rows = frame[frame.run_id == run["run_id"]]
        if rows.empty or set(rows.label) != {run["condition"]} or len(rows) != len(run["windows"]):
            raise ValueError("run/label/window mismatch")
        if not np.array_equal(rows.window_id.to_numpy(), np.arange(len(rows))):
            raise ValueError("window sequence mismatch")
        decoded = pd.DataFrame(run["telemetry"])
        if not np.allclose(np.diff(decoded.timestamp_s), 0.1, atol=1e-12):
            raise ValueError("ECU sampling mismatch")
        if not np.allclose(decoded.engine_load * 100, decoded.engine_load_pct, atol=1e-12):
            raise ValueError("load unit mismatch")
        event_times = np.array([t for t, _ in run["ckp_events"]])
        if np.any(np.diff(event_times) <= 0):
            raise ValueError("CKP timestamp order mismatch")
        tick = manifest["configuration"]["timestamp_tick_s"]
        if not np.allclose(event_times / tick, np.round(event_times / tick), atol=1e-8, rtol=0):
            raise ValueError("CKP clock quantization mismatch")
        for row in rows.to_dict(orient="records"):
            start, end = row["t_start"], row["t_end"]
            if (abs(start - row["window_id"] * 0.25) > 1e-12
                    or abs(end - start - 0.5) > 1e-12):
                raise ValueError("window alignment mismatch")
            events = [(t, i) for t, i in run["ckp_events"] if start <= t < end]
            ckp = process_edges(events, start=start, end=end, timestamp_resolution_s=tick)
            if ckp["quality_status"] != "good":
                raise ValueError("bad acquisition evidence")
            minimum_resolution = min(minimum_resolution, ckp["features"]["ckp_min_samples_per_tooth"])
            for name, value in ckp["features"].items():
                if not np.isclose(row[name], value, atol=1e-10, rtol=1e-10):
                    raise ValueError(f"CKP feature/evidence mismatch: {name}")
            for family in ("map_kpa", "engine_load_pct"):
                result = window_features(list(zip(decoded.timestamp_s, decoded[family])),
                                         start=start, end=end, family=family)
                for name, value in result["features"].items():
                    if value is None or not np.isclose(row[name], value, atol=1e-10, rtol=1e-10):
                        raise ValueError(f"ECU feature/evidence mismatch: {name}")
    report["minimum_timestamp_ticks_per_tooth"] = minimum_resolution
    report["source_hashes_and_saved_evidence_verified"] = True
    # Verify future evaluation splits only; do not instantiate or fit a classifier.
    report["five_fold_group_class_coverage"] = "not checked: smoke dataset"
    if min(report["runs_per_class"].values()) >= 5:
        for train, test in GroupKFold(5).split(frame, frame.label, frame.run_id):
            if (set(frame.iloc[train].run_id) & set(frame.iloc[test].run_id)
                    or set(frame.iloc[train].label) != set(CONDITIONS)
                    or set(frame.iloc[test].label) != set(CONDITIONS)):
                raise ValueError("invalid future grouped evaluation split")
        report["five_fold_group_class_coverage"] = "PASS; no estimator fitted"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_directory(args.directory), indent=2, allow_nan=False))
