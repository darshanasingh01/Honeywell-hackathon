"""Focused tests for the NUST recording-level bearing-condition baseline."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ml.diagnostics.adapters.vibration import wrap_vibration_output
from ml.diagnostics.schema import validate_prediction
from ml.vibration.bearing_condition import (
    CHANNEL, FEATURE_NAMES, FAULTY_LABEL, HEALTHY_LABEL,
    discover_recordings, extract_time_domain_features, load_recording,
)
from ml.vibration.predict_bearing_condition import predict_recording
from ml.vibration.train_bearing_condition import stratified_recording_folds


HEADERS = [
    "Time", "Demand 1", "Control 1", "Output Drive 1",
    "Channel 1", "Channel 2", "Channel 3", "Channel 4",
    "Channel 1 Kurtosis", "Channel 2 Kurtosis", "Channel 3 Kurtosis", "Channel 4 Kurtosis",
    "Rear Input 1", "Rear Input 2", "Rear Input 3", "Rear Input 4",
    "Rear Input 5", "Rear Input 6", "Rear Input 7", "Rear Input 8",
]


def write_recording(path: Path, *, channel3=None, count: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channel3 = channel3 or [1.0 + math.sin(i * 0.37) + i * 0.002 for i in range(count)]
    rows = []
    for i, sample in enumerate(channel3):
        rows.append({
            "Time": f"{0.1 + i * 0.0034:.7f}", "Demand 1": "0.1", "Control 1": "0.2",
            "Output Drive 1": "0", "Channel 1": f"{0.2 + sample:.7f}",
            "Channel 2": f"{0.3 + sample:.7f}", CHANNEL: f"{sample:.7f}",
            "Channel 4": "0", "Channel 1 Kurtosis": "2", "Channel 2 Kurtosis": "2",
            "Channel 3 Kurtosis": "2", "Channel 4 Kurtosis": "2",
            **{f"Rear Input {j}": "0" for j in range(1, 9)},
        })
    rows.insert(1, dict(rows[0]))  # exact repeated initial row present in the downloaded files
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


class BearingConditionTests(unittest.TestCase):
    def test_csv_channel_selection_and_initial_duplicate_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Healthy Bearings Dataset" / "capture2022May27.csv"
            write_recording(path)
            recording = load_recording(path)
            self.assertEqual(recording.label, HEALTHY_LABEL)
            self.assertTrue(recording.removed_initial_duplicate)
            self.assertEqual(recording.samples.size, 32)
            self.assertEqual(recording.timestamp_quality["timestamp_repeats"], 0)
            self.assertEqual(recording.timestamp_quality["sample_rate_verified"], False)

    def test_byte_identical_recordings_are_deduplicated_before_splitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "Healthy Bearings Dataset" / "capture2022May27.csv"
            duplicate = root / "Healthy Bearings Dataset" / "copy2022May27.csv"
            write_recording(original)
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            duplicate.write_bytes(original.read_bytes())
            recordings, counts = discover_recordings(root)
            self.assertEqual(len(recordings), 1)
            self.assertEqual(counts, {"discovered_csv_files": 2, "deduplicated_files": 1})

    def test_time_domain_features_are_finite_and_sample_rate_independent(self):
        feature_values = extract_time_domain_features(np.asarray([1.0 + math.sin(i) for i in range(32)]))
        self.assertEqual(list(feature_values), FEATURE_NAMES)
        self.assertTrue(all(math.isfinite(value) for value in feature_values.values()))
        self.assertFalse(any("hz" in name.lower() or "fft" in name.lower() for name in feature_values))
        flat = extract_time_domain_features(np.ones(32))
        self.assertTrue(all(math.isfinite(value) for value in flat.values()))
        self.assertEqual(flat["crest_factor"], 0.0)

    def test_recording_level_stratified_folds_hold_out_each_record_once(self):
        labels = np.asarray([HEALTHY_LABEL] * 10 + [FAULTY_LABEL] * 15, dtype=object)
        held_out = []
        for train_indices, test_indices in stratified_recording_folds(labels, folds=5, seed=9):
            self.assertFalse(set(train_indices) & set(test_indices))
            self.assertGreater(len(set(labels[test_indices])), 1)
            held_out.extend(test_indices.tolist())
        self.assertEqual(sorted(held_out), list(range(len(labels))))

    def test_prediction_uses_channel_and_returns_uncalibrated_probs(self):
        """A stub model isolates input, feature, and response-contract behavior."""
        class StubModel:
            classes_ = np.asarray([FAULTY_LABEL, HEALTHY_LABEL], dtype=object)

            def predict(self, frame):
                return np.asarray([HEALTHY_LABEL])

            def predict_proba(self, frame):
                return np.asarray([[0.2, 0.8]])

        from ml.vibration import predict_bearing_condition as prediction_module
        with tempfile.TemporaryDirectory() as temporary:
            signal = Path(temporary) / "sample.csv"
            write_recording(signal)
            bundle = {"model_version": "engineguard-vibration-nust-bearing-condition-v1",
                      "input_channel": CHANNEL, "feature_names": FEATURE_NAMES,
                      "model": StubModel(), "limitations": []}
            with patch.object(prediction_module.joblib, "load", return_value=bundle):
                output = predict_recording(signal, "unused.joblib")
        self.assertEqual(output["predicted_bearing_condition"], HEALTHY_LABEL)
        self.assertEqual(output["class_probabilities_uncalibrated"][HEALTHY_LABEL], 0.8)
        self.assertEqual(output["input_channel"], CHANNEL)
        self.assertFalse(output["probabilities_calibrated"])

    def test_adapter_is_shared_schema_compatible_without_invented_provenance(self):
        native = {
            "predicted_bearing_condition": FAULTY_LABEL,
            "class_probabilities_uncalibrated": {HEALTHY_LABEL: 0.25, FAULTY_LABEL: 0.75},
            "model_version": "nust-test-v1", "input_channel": CHANNEL,
            "signal_quality": {"status": "poor", "metrics": {"sample_rate_verified": False}},
            "limitations": ["Uncalibrated scores."],
        }
        result = wrap_vibration_output(native, prediction_id="test-prediction")
        validate_prediction(result)
        self.assertEqual(result["sensor_type"], "vibration")
        self.assertEqual(result["fault_prediction"], FAULTY_LABEL)
        self.assertIsNone(result["engine_id"])
        self.assertIsNone(result["event_id"])
        self.assertEqual(result["signal_quality"]["status"], "poor")

    def test_missing_channel_and_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / "Faulty Bearings Dataset" / "bad.csv"
            write_recording(bad)
            content = bad.read_text(encoding="utf-8")
            bad.write_text(content.replace("Channel 3,", "Missing Channel,"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_recording(bad)
            write_recording(bad)
            content = bad.read_text(encoding="utf-8").replace("1.0000000", "NaN", 1)
            bad.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite numeric"):
                load_recording(bad)


if __name__ == "__main__":
    unittest.main()
