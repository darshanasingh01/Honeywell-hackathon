import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.real_data_dashboard_adapter import RealDataAdapter


def make_exports(root: Path) -> None:
    pd.DataFrame([
        {"recording_key": "rec1", "rec_num": 1, "fuel": "diesel", "label": 0, "path": "rec1.wav", "split": "train"},
        {"recording_key": "rec2", "rec_num": 2, "fuel": "petrol", "label": 1, "path": "rec2.wav", "split": "test"},
    ]).to_csv(root / "real_audio_manifest.csv", index=False)
    pd.DataFrame([
        {"window_id": "rec1_w000", "recording_key": "rec1", "rec_num": 1, "fuel": "diesel", "label": 0, "split": "train", "start_s": 0.0, "end_s": 0.5, "padded": 0, "clipped": 0, "rms": 1.0},
        {"window_id": "rec2_w000", "recording_key": "rec2", "rec_num": 2, "fuel": "petrol", "label": 1, "split": "test", "start_s": 0.0, "end_s": 0.5, "padded": 0, "clipped": 0, "rms": 2.0},
    ]).to_csv(root / "real_audio_windows.csv", index=False)
    pd.DataFrame([
        {"window_id": "csv_w00000", "chunk_id": 0, "run_id": 0, "split": "train", "row_start": 128, "row_end": 256, "fault": 0, "rpm_mean": 800.0},
        {"window_id": "csv_w00001", "chunk_id": 1, "run_id": 0, "split": "test", "row_start": 2128, "row_end": 2256, "fault": 3, "rpm_mean": 900.0},
    ]).to_csv(root / "real_csv_windows.csv", index=False)
    np.savez_compressed(root / "real_audio_spectrograms.npz",
                        spectrograms=np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3),
                        window_id=np.array(["rec1_w000", "rec2_w000"], dtype=object))
    (root / "real_audio_summary.json").write_text(json.dumps({"n_recordings": 2, "n_windows": 2}), encoding="utf-8")
    (root / "real_csv_summary.json").write_text(json.dumps({"window_report": {"n_windows": 2}}), encoding="utf-8")


class RealDataDashboardAdapterTests(unittest.TestCase):
    def test_labels_splits_and_spectrogram_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_exports(root)
            adapter = RealDataAdapter(root)
            self.assertEqual({r["fuel"] for r in adapter.list_audio_recordings()}, {"diesel", "petrol"})
            row = adapter.audio_windows("rec1")[0]
            result = adapter.audio_spectrogram(row["window_id"])
            self.assertEqual(result["feature_row"]["window_id"], row["window_id"])
            self.assertEqual(result["spectrogram"].shape, (2, 3))
            self.assertEqual(adapter.list_csv_fault_codes(), [0, 3])
            self.assertEqual(adapter.list_csv_splits(), ["test", "train"])
            self.assertEqual(adapter.csv_windows(split="test")[0]["row_start"], 2128)
            self.assertEqual(adapter.metadata()["audio"]["fault_label"], "unknown")

    def test_rejects_spectrogram_misalignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_exports(root)
            np.savez_compressed(root / "real_audio_spectrograms.npz",
                                spectrograms=np.zeros((1, 2, 3)), window_id=np.array(["missing"], dtype=object))
            with self.assertRaisesRegex(ValueError, "spectrogram IDs"):
                RealDataAdapter(root).metadata()

    def test_reports_missing_artifact(self):
        with self.assertRaisesRegex(FileNotFoundError, "real-data export directory"):
            RealDataAdapter(Path("does-not-exist"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_exports(root)
            (root / "real_csv_windows.csv").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "real_csv_windows.csv"):
                RealDataAdapter(root).csv_windows()


if __name__ == "__main__":
    unittest.main()
