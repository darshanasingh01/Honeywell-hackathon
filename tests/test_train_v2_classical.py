"""Focused tests for separate V2 SVM and softmax artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from ml.synthetic_conditions.predict_v2_classical import load_model, predict_one
from ml.synthetic_conditions.schema_v2 import FEATURE_COLUMNS, MANIFEST
from ml.synthetic_conditions.train_v2_classical import (
    DEFAULT_SOFTMAX_MODEL, DEFAULT_SVM_MODEL, DEFAULT_REPORT,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "ml/synthetic_conditions/v2_examples.json"


class ClassicalV2Tests(unittest.TestCase):
    @unittest.skipUnless(DEFAULT_SVM_MODEL.exists() and DEFAULT_SOFTMAX_MODEL.exists(),
                         "classical V2 model artifacts are absent")
    def test_loading_and_label_order(self):
        for path, version in ((DEFAULT_SVM_MODEL, "synthetic-conditions-167-v2-svm-v1"),
                              (DEFAULT_SOFTMAX_MODEL, "synthetic-conditions-167-v2-softmax-v1")):
            bundle = load_model(path)
            self.assertEqual(bundle["model_version"], version)
            self.assertEqual(bundle["feature_columns"], list(FEATURE_COLUMNS))
            self.assertEqual(bundle["class_labels"], list(MANIFEST["class_labels"]))

    @unittest.skipUnless(DEFAULT_SVM_MODEL.exists() and DEFAULT_SOFTMAX_MODEL.exists(),
                         "classical V2 model artifacts are absent")
    def test_inference_semantics_and_examples(self):
        examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))
        for path, semantics in ((DEFAULT_SVM_MODEL, "svm_decision_score_not_probability"),
                                (DEFAULT_SOFTMAX_MODEL, "softmax_model_probability_not_calibrated")):
            bundle = load_model(path)
            for expected in ("normal", "misfire", "overheating"):
                result = predict_one(bundle, examples[expected])
                self.assertEqual(result["predicted_condition"], expected)
                self.assertEqual(result["class_order"], list(MANIFEST["class_labels"]))
                self.assertEqual(list(result["condition_scores"]), list(MANIFEST["class_labels"]))
                self.assertEqual(result["score_semantics"], semantics)
            if "softmax" in bundle["model_version"]:
                scores = predict_one(bundle, examples["normal"])["condition_scores"].values()
                self.assertAlmostEqual(sum(scores), 1.0, places=6)

    @unittest.skipUnless(DEFAULT_SVM_MODEL.exists() and DEFAULT_SOFTMAX_MODEL.exists(),
                         "classical V2 model artifacts are absent")
    def test_strict_feature_contract(self):
        values = json.loads(EXAMPLES.read_text(encoding="utf-8"))["normal"]
        bundle = load_model(DEFAULT_SVM_MODEL)
        for invalid in (dict(values, extra_feature=1),
                        dict(values, mic_std="1"),
                        dict(values, mic_std=True),
                        dict(values, mic_std=float("nan"))):
            with self.assertRaises(ValueError):
                predict_one(bundle, invalid)

    def test_report_exists_after_training(self):
        self.assertTrue(DEFAULT_REPORT.exists(), "run train_v2_classical.py to create the report")
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["dataset"]["feature_count"], 167)
        self.assertEqual(report["evaluation"]["class_order"], list(MANIFEST["class_labels"]))
        self.assertTrue(report["evaluation"]["preprocessing_fitted_within_training_fold"])
        self.assertEqual(set(report["models"]), {"scaled_rbf_svm", "standardized_softmax"})


if __name__ == "__main__":
    unittest.main()
