import json
from pathlib import Path
import tempfile
import wave

import numpy as np
import unittest

from ml.acoustic.real_audio_audit import audit_wavs, export


def write_wav(path, samples, rate=8000):
    values = np.asarray(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate)
        stream.writeframes(values.tobytes())


class RealAudioAuditTests(unittest.TestCase):
    def test_missing_requested_root_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            result = export(Path(temp) / "missing", Path(temp) / "out")
            self.assertEqual(result["status"], "source_missing")
            self.assertEqual(result["wav_count"], 0)

    def test_manifest_features_duplicates_clipping_and_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / "raw", Path(temp) / "export"
            root.mkdir()
            t = np.arange(16000) / 8000
            write_wav(root / "a.wav", (12000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16))
            (root / "duplicate.wav").write_bytes((root / "a.wav").read_bytes())
            write_wav(root / "clipped.wav", np.r_[np.zeros(1000), np.full(1000, 32767)])
            records, summary = audit_wavs(root)
            self.assertEqual(summary["wav_count"], 3)
            self.assertEqual(summary["readable_count"], 3)
            self.assertEqual(summary["duplicate_group_count"], 1)
            self.assertGreater(summary["clipped_file_count"], 0)
            result = export(root, output)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(result["sample_exports"]), 3)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(all(r["label"] == "unknown" for r in manifest["records"]))
            self.assertFalse(manifest["feature_provenance"]["mfcc_available"])
            for sample in result["sample_exports"]:
                self.assertTrue((output / sample["filename"]).exists())


if __name__ == "__main__":
    unittest.main()
