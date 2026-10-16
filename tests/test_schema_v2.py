"""Analytical contract fixtures, not generated engine-condition training data."""
import copy
import unittest

import numpy as np

from ml.synthetic_conditions.ckp_v2 import process_edges
from ml.synthetic_conditions.ecu_v2 import decode_pid, window_features
from ml.synthetic_conditions.schema_v2 import MANIFEST, FEATURE_COLUMNS, validate_row
from ml.synthetic_conditions.schema_136 import FEATURE_COLUMNS as V1


def edges(irregular=False):
    result = []
    t = 0.0
    previous = 0
    for revolution in range(7):
        for tooth in range(58):
            position = revolution * 60 + tooth
            t += (position - previous) / 1500 * (1 + (0.12 * np.sin(position / 9) if irregular else 0))
            result.append((t, tooth))
            previous = position
    return result


class V2Tests(unittest.TestCase):
    def ckp(self, events, resolution=1e-5):
        return process_edges(events, start=0, end=0.28, timestamp_resolution_s=resolution)

    def test_healthy_and_irregular_edges(self):
        healthy = self.ckp(edges())
        irregular = self.ckp(edges(True))
        self.assertEqual(healthy["quality_status"], "good")
        self.assertAlmostEqual(healthy["features"]["ckp_tooth_rpm_mean"], 1500)
        self.assertEqual(irregular["quality_status"], "good")
        self.assertGreater(irregular["features"]["ckp_angular_speed_cv"], 0.05)
        self.assertEqual(healthy["features"]["ckp_unexpected_gap_count"], 0)

    def test_missing_and_undersampled(self):
        lost = edges(); del lost[30]
        for result in (self.ckp(lost), self.ckp(edges(), 0.001), self.ckp([]), self.ckp(None)):
            self.assertNotEqual(result["quality_status"], "good")
            self.assertIsNone(result["features"]["ckp_tooth_rpm_mean"])
        self.assertEqual(self.ckp(lost)["features"]["ckp_unexpected_gap_count"], 1)

    def test_bad_timestamps_indices_and_rpm(self):
        for events_ in ([(0, 0), (0, 1)], [(0, 58)], [(0, True)]):
            with self.assertRaises(ValueError):
                self.ckp(events_)
        self.assertEqual(self.ckp([(t / 10, i) for t, i in edges()])["quality_status"], "poor_quality")

    def test_pid_units_and_strict_bytes(self):
        self.assertEqual(decode_pid(0x0B, [100]), {"map_kpa": 100.0})
        self.assertEqual(decode_pid(0x04, [255]), {"engine_load": 1.0, "engine_load_pct": 100.0})
        for payload in ([True], [256], [1, 2], [1.0]):
            with self.assertRaises(ValueError):
                decode_pid(0x0B, payload)

    def test_ecu_boundaries_missing_stale_and_zero(self):
        samples = [(0, 0), (0.1, 10), (0.2, 20), (0.3, 255)]
        result = window_features(samples, start=0, end=0.3, family="map_kpa")
        self.assertEqual(result["quality_status"], "good")
        self.assertEqual(result["features"]["tel_map_kpa_min"], 0)
        self.assertAlmostEqual(result["features"]["tel_map_kpa_slope"], 100)
        for data, status in ((None, "unavailable"), ([], "insufficient_samples"),
                             ([(0, 1), (0.5, 2)], "stale")):
            self.assertEqual(window_features(data, start=0, end=0.6, family="map_kpa")["quality_status"], status)
        for data in ([(0, 256), (0.1, 10)], [(0, 1), (0, 2)], [(0, "1")]):
            with self.assertRaises(ValueError):
                window_features(data, start=0, end=0.3, family="map_kpa")

    def test_contract_and_quality_missingness(self):
        self.assertEqual(FEATURE_COLUMNS[:136], V1)
        self.assertEqual(len(set(FEATURE_COLUMNS)), 167)
        row = dict.fromkeys(FEATURE_COLUMNS)
        row.update(run_id="analytical", window_id=0, label="normal", t_start=0, t_end=1,
                   source_id="unit-test-only", source_commit="analytical-fixture",
                   processor_revision="unit-test", data_kind="synthetic_simulated_engine_conditions",
                   timestamp_resolution_s=1e-5)
        for group in MANIFEST["sensor_groups"]:
            row["available_" + group["name"]] = False
        for source in ("ckp", "map", "engine_load"):
            row[source + "_signal_quality_status"] = "unavailable"
        validate_row(row)
        for change in ({"tel_map_kpa_mean": 0}, {"available_map": True}, {"source_id": ""},
                       {"timestamp_resolution_s": True}, {"extra": 1}):
            changed = copy.deepcopy(row); changed.update(change)
            with self.assertRaises(ValueError):
                validate_row(changed)


if __name__ == "__main__":
    unittest.main()
