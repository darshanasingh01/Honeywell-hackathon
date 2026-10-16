"""Validation for simulator-derived V2 fast-sensor spectrogram exports."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "ml/synthetic_conditions/fast_spectrograms_v2_complete_retry"


class FastSpectrogramV2Tests(unittest.TestCase):
    def test_five_conditions_arrays_and_alignment(self):
        index = json.loads((EXPORT / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(set(index["exports"]),
                         {"normal", "knock", "misfire", "overheating", "valve_train_wear"})
        self.assertEqual(index["sample_rate_hz"], 25600)
        for condition, item in index["exports"].items():
            path = EXPORT / item["path"]
            self.assertTrue(path.is_file())
            with np.load(path, allow_pickle=False) as archive:
                frequencies = archive["frequencies_hz"]
                times = archive["times_s"]
                metadata = json.loads(str(archive["metadata_json"]))
                self.assertEqual(metadata["condition_label"], condition)
                self.assertEqual(metadata["run_id"], item["run_id"])
                self.assertEqual(metadata["window_id"], 0)
                self.assertTrue(metadata["alignment"]["same_v2_window"])
                self.assertLessEqual(metadata["alignment"]["feature_match_max_abs_error"], 1e-8)
                self.assertTrue(np.all(np.isfinite(frequencies)))
                self.assertTrue(np.all(np.isfinite(times)))
                start, end = metadata["window"]["t_start"], metadata["window"]["t_end"]
                self.assertTrue(np.all((times >= start) & (times < end)))
                for channel in index["channels"]:
                    array = archive[channel + "_magnitude"]
                    self.assertEqual(array.shape, tuple(metadata["array_shapes"][channel]))
                    self.assertEqual(array.shape[1], times.size)
                    self.assertTrue(np.all(np.isfinite(array)))


if __name__ == "__main__":
    unittest.main()
