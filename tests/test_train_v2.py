import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ml.synthetic_conditions.predict_v2 import load_model, predict_one
from ml.synthetic_conditions.schema_v2 import FEATURE_COLUMNS, MANIFEST
from ml.synthetic_conditions.train_v2 import (
    DEFAULT_DATASET, DEFAULT_MODEL, DEFAULT_REPORT, evaluate, load_training_handoff,
)


class TrainV2Tests(unittest.TestCase):
    def test_handoff_and_grouped_evaluation(self):
        frame, X, y, groups, manifest, verification = load_training_handoff(DEFAULT_DATASET)
        self.assertEqual((len(frame), X.shape[1], groups.nunique()), (1170, 167, 30))
        self.assertTrue(verification["source_hashes_and_saved_evidence_verified"])
        results, selected_name, selected = evaluate(X, y, groups)
        self.assertEqual(selected_name, "random_forest")
        for result in results.values():
            self.assertEqual(len(result["folds"]), 5)
            self.assertEqual(set(result["out_of_fold_metrics"]["per_class_recall"]), set(MANIFEST["class_labels"]))
            self.assertTrue(all(not set(f["train_run_ids"]) & set(f["validation_run_ids"]) for f in result["folds"]))
        self.assertAlmostEqual(results["random_forest"]["mean_fold_macro_f1"], 1.0)

    def test_bundle_and_inference_contract(self):
        bundle = load_model(DEFAULT_MODEL)
        self.assertEqual(bundle["feature_columns"], list(FEATURE_COLUMNS))
        row = json.loads(Path("tests/fixtures/synthetic_condition_136_smoke.json").read_text())
        # The V2 fixture is taken from a validated training row, not padded V1 data.
        import pandas as pd
        v2 = pd.read_csv(DEFAULT_DATASET / "features.csv", nrows=1).iloc[0]
        values = {name: v2[name].item() if hasattr(v2[name], "item") else v2[name] for name in FEATURE_COLUMNS}
        output = predict_one(bundle, values)
        self.assertIn(output["predicted_condition"], bundle["class_labels"])
        self.assertEqual(set(output["condition_scores"]), set(bundle["class_labels"]))
        self.assertFalse(output["dataset_provenance"]["real_engine_measurements"])
        for invalid in (dict(values, extra=1), dict(values, mic_std="1"), dict(values, mic_std=True),
                        dict(values, mic_std=float("nan"))):
            with self.assertRaises(ValueError):
                predict_one(bundle, invalid)

    def test_v1_artifact_paths_are_separate(self):
        self.assertNotEqual(DEFAULT_MODEL.name, "synthetic_conditions_136_baseline.joblib")
        self.assertNotEqual(DEFAULT_REPORT.name, "synthetic_conditions_136_groupkfold.json")


if __name__ == "__main__":
    unittest.main()
