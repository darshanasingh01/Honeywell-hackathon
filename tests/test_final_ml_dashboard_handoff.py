"""Read-only verification for the final ML dashboard handoff."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from ml.synthetic_conditions.predict_v2 import load_model, predict_one
from ml.synthetic_conditions.schema_v2 import FEATURE_COLUMNS, MANIFEST
from ml.synthetic_conditions.train_v2 import DEFAULT_MODEL


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_INDEX = ROOT / "ml/synthetic_conditions/sensor_exports_v2/index.json"
REAL_INDEX = ROOT / "ml/acoustic/real_audio_exports_v2/index.json"
REAL_ROOT = REAL_INDEX.parent
EXAMPLES = ROOT / "ml/synthetic_conditions/v2_examples.json"
EXPECTED_MODEL_SHA256 = "01f693c753d3439f0eb4b7660aac4d95ab339529d4f5a53bc4435c2e11966db3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FinalDashboardHandoffTests(unittest.TestCase):
    @unittest.skipUnless(DEFAULT_MODEL.exists(), "local ignored V2 model artifact is absent")
    def test_model_and_examples(self):
        self.assertEqual(sha256(DEFAULT_MODEL), EXPECTED_MODEL_SHA256)
        bundle = load_model(DEFAULT_MODEL)
        self.assertEqual(bundle["feature_columns"], list(FEATURE_COLUMNS))
        self.assertEqual(bundle["class_labels"], list(MANIFEST["class_labels"]))
        examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))
        for expected in ("normal", "misfire", "overheating"):
            values = examples[expected]
            self.assertEqual(set(values), set(FEATURE_COLUMNS))
            result = predict_one(bundle, values)
            self.assertEqual(result["predicted_condition"], expected)
            self.assertEqual(set(result["condition_scores"]), set(MANIFEST["class_labels"]))
            self.assertEqual(result["score_semantics"], "random_forest_vote_fraction_uncalibrated")

    def test_referenced_exports_exist_and_match_hashes(self):
        synthetic = json.loads(SYNTHETIC_INDEX.read_text(encoding="utf-8"))
        self.assertFalse(synthetic["real_engine_measurements"])
        for item in synthetic["exports"].values():
            path = SYNTHETIC_INDEX.parent / item["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(sha256(path), item["sha256"])

        real = json.loads(REAL_INDEX.read_text(encoding="utf-8"))
        self.assertEqual(real["summary"]["wav_count"], 152)
        self.assertEqual(real["label_policy"], "unknown unless verified metadata exists; no fault labels inferred")
        self.assertTrue((REAL_ROOT / "manifest.json").is_file())
        self.assertEqual(sha256(REAL_ROOT / "manifest.json"), real["manifest_sha256"])
        for item in real["sample_exports"]:
            path = REAL_ROOT / item["filename"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(sha256(path), item["sha256"])


if __name__ == "__main__":
    unittest.main()
