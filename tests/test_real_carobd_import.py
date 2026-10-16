"""Validation for the isolated real carOBD telemetry import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ml/datasets/real_obd_carobd_v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RealCarobdImportTests(unittest.TestCase):
    def test_manifest_and_dashboard_are_real_unlabeled_data(self):
        manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        dashboard = json.loads((DATA / "dashboard_timeseries.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["data_kind"], "real_recorded_ecu_telemetry")
        self.assertEqual(manifest["diagnostic_label"], "unlabeled")
        self.assertEqual(manifest["exported_row_count"], 240)
        self.assertEqual(set(dashboard["units"]), {
            "engine_rpm", "engine_load_percent", "map_kpa", "throttle_percent",
            "coolant_temperature_degC", "short_term_fuel_trim_percent",
            "long_term_fuel_trim_percent", "ignition_timing_advance_degBTDC",
        })
        self.assertEqual(len(dashboard["trajectories"]), 2)
        for trajectory in dashboard["trajectories"]:
            timestamps = [row["timestamp_s"] for row in trajectory["records"]]
            self.assertEqual(timestamps, sorted(timestamps))
            self.assertEqual(len(trajectory["records"]), 120)
            for row in trajectory["records"]:
                self.assertEqual(set(row["mapped"]), set(dashboard["units"]))
                self.assertIn("source_values", row)

    def test_raw_files_and_hash_provenance(self):
        manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        for name, trajectory in zip(("idle1.csv", "drive1.csv"), manifest["trajectories"]):
            path = DATA / "raw" / name
            self.assertTrue(path.is_file())
            self.assertEqual(digest(path), trajectory["source_sha256"])
            self.assertGreater(trajectory["source_row_count"], trajectory["exported_row_count"])


if __name__ == "__main__":
    unittest.main()
