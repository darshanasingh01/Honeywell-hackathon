"""Leakage and inference-contract tests for synthetic condition baselines."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from ml.synthetic_conditions.predict import load_model, predict_one
from ml.synthetic_conditions.train import (
    CLASS_ORDER,
    META_COLUMNS,
    grouped_splits,
    prepare_xy,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for class_index, label in enumerate(CLASS_ORDER):
        for run_number in range(6):
            run_id = f"{label}_s{run_number}"
            rows.append({
                "run_id": run_id,
                "window_id": 0,
                "label": label,
                "t_start": 0.0,
                "t_end": 0.5,
                **{f"feature_{i}": float(class_index + run_number + i / 100.0) for i in range(74)},
            })
    return pd.DataFrame(rows)


class DummyEstimator:
    classes_ = list(CLASS_ORDER)

    def predict(self, X):
        return ["normal"]

    def predict_proba(self, X):
        return [[0.8, 0.05, 0.05, 0.05, 0.05]]


class SyntheticConditionTests(unittest.TestCase):
    def test_metadata_excluded_from_model_features(self):
        frame = sample_frame()
        X, y, groups = prepare_xy(frame)
        self.assertEqual(X.shape, (30, 74))
        self.assertFalse(set(META_COLUMNS) & set(X.columns))
        self.assertEqual(y.name, "label")
        self.assertEqual(groups.name, "run_id")

    def test_groupkfold_keeps_each_run_out_of_train_test_overlap(self):
        frame = sample_frame()
        X, y, groups = prepare_xy(frame)
        folds = list(grouped_splits(X, y, groups))
        self.assertEqual(len(folds), 5)
        for train_idx, test_idx in folds:
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            self.assertFalse(train_groups & test_groups)
            self.assertEqual(set(y.iloc[test_idx]), set(CLASS_ORDER))

    def test_inference_output_contract_marks_simulated_origin(self):
        frame = sample_frame()
        X, _, _ = prepare_xy(frame)
        bundle = {
            "model": DummyEstimator(),
            "model_name": "random_forest",
            "model_version": "synthetic-engine-condition-baseline-v1",
            "feature_columns": list(X.columns),
        }
        prediction = predict_one(bundle, X.iloc[0].to_dict())
        self.assertEqual(prediction["predicted_condition"], "normal")
        self.assertEqual(set(prediction["condition_scores"]), set(CLASS_ORDER))
        self.assertEqual(prediction["score_semantics"], "random_forest_vote_fraction_uncalibrated")
        self.assertTrue(prediction["dataset_provenance"]["data_kind"].startswith("synthetic_"))
        self.assertFalse(prediction["dataset_provenance"]["real_engine_measurements"])
        self.assertTrue(any("not a validated real-world" in x for x in prediction["limitations"]))

    def test_inference_rejects_metadata_as_feature_input(self):
        frame = sample_frame()
        X, _, _ = prepare_xy(frame)
        bundle = {
            "model": DummyEstimator(),
            "model_name": "random_forest",
            "model_version": "test-v1",
            "feature_columns": list(X.columns),
        }
        row = X.iloc[0].to_dict()
        row["run_id"] = "normal_s0"
        with self.assertRaisesRegex(ValueError, "metadata must not be supplied"):
            predict_one(bundle, row)

    def test_missing_model_artifact_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.joblib"
            with self.assertRaisesRegex(FileNotFoundError, "Train it from the repository root"):
                load_model(missing)

    def test_svm_decision_scores_are_not_probabilities(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        frame = sample_frame()
        X, y, _ = prepare_xy(frame)
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=False)),
        ]).fit(X, y)
        bundle = {
            "model": estimator,
            "model_name": "scaled_rbf_svm",
            "model_version": "test-v1",
            "feature_columns": list(X.columns),
        }
        prediction = predict_one(bundle, X.iloc[0].to_dict())
        self.assertEqual(prediction["score_semantics"], "svm_decision_score_not_probability")
        self.assertNotIn("probabilities", prediction)


if __name__ == "__main__":
    unittest.main()
