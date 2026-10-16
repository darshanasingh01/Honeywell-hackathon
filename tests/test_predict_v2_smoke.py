"""Standalone dashboard handoff smoke test for the saved V2 artifact."""
import json
from pathlib import Path
import unittest

from ml.synthetic_conditions.predict_v2 import load_model, predict_one
from ml.synthetic_conditions.schema_v2 import FEATURE_COLUMNS, MANIFEST
from ml.synthetic_conditions.train_v2 import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "ml" / "synthetic_conditions" / "v2_examples.json"


class PredictV2SmokeTests(unittest.TestCase):
    @unittest.skipUnless(DEFAULT_MODEL.exists(), "local ignored V2 model artifact is absent")
    def test_three_validated_examples(self):
        bundle = load_model(DEFAULT_MODEL)
        examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))
        self.assertEqual(set(examples), {"normal", "misfire", "overheating"})
        for expected, values in examples.items():
            self.assertEqual(set(values), set(FEATURE_COLUMNS))
            result = predict_one(bundle, values)
            self.assertEqual(result["predicted_condition"], expected)
            self.assertEqual(set(result["condition_scores"]), set(MANIFEST["class_labels"]))
            self.assertEqual(result["score_semantics"], "random_forest_vote_fraction_uncalibrated")
            self.assertFalse(result["dataset_provenance"]["real_engine_measurements"])

    @unittest.skipUnless(DEFAULT_MODEL.exists(), "local ignored V2 model artifact is absent")
    def test_contract_errors_are_actionable(self):
        bundle = load_model(DEFAULT_MODEL)
        values = json.loads(EXAMPLES.read_text(encoding="utf-8"))["normal"]
        for invalid in (dict(values, extra_feature=1),
                        dict(values, mic_std="1"),
                        dict(values, mic_std=True),
                        dict(values, mic_std=float("nan"))):
            with self.assertRaises(ValueError):
                predict_one(bundle, invalid)


if __name__ == "__main__":
    unittest.main()
