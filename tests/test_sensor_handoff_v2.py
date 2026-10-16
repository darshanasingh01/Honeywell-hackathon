import json
from pathlib import Path
import tempfile
import unittest

from ml.synthetic_conditions.export_sensor_v2 import (
    DEFAULT_DATASET, REQUESTED_CONDITIONS, SENSOR_GROUPS, export_samples,
)


class SensorHandoffV2Tests(unittest.TestCase):
    def test_export_has_all_groups_and_same_window_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "export"
            index = export_samples(DEFAULT_DATASET, output)
            self.assertEqual(index["sensor_group_count"], 10)
            self.assertEqual(index["primary_sensor_group_count"], 9)
            self.assertEqual(set(index["exports"]), set(REQUESTED_CONDITIONS))
            self.assertEqual(index["missing_evidence"]["raw_microphone_waveform"], True)
            for condition in REQUESTED_CONDITIONS:
                payload = json.loads((output / f"{condition}_window0.json").read_text())
                self.assertEqual(payload["condition_label"], condition)
                self.assertEqual(payload["window_id"], 0)
                self.assertEqual(set(payload["sensor_groups"]), set(SENSOR_GROUPS))
                self.assertTrue(payload["synchronization"]["same_generated_run"])
                self.assertTrue(payload["synchronization"]["same_window"])
                self.assertTrue(all(payload["window"]["t_start"] <= x["timestamp_s"] < payload["window"]["t_end"]
                                    for x in payload["telemetry_samples"] + payload["ckp_events"]))
                self.assertFalse(payload["waveform_evidence"]["raw_waveform_available"])
                self.assertTrue(payload["waveform_evidence"]["feature_summary_available"])
                self.assertEqual(payload["provenance"]["source_id"], payload["run_id"])
                self.assertEqual(payload["provenance"]["source_id"].split("_s")[0].replace("v2_train_", ""), condition)
            with self.assertRaises(FileExistsError):
                export_samples(DEFAULT_DATASET, output)

    def test_no_unrequested_or_fabricated_sensor_data(self):
        with tempfile.TemporaryDirectory() as temp:
            index = export_samples(DEFAULT_DATASET, Path(temp) / "export")
            for name in ("raw_microphone_waveform", "raw_knock_waveform", "raw_accelerometer_waveform"):
                self.assertTrue(index["missing_evidence"][name])


if __name__ == "__main__":
    unittest.main()
