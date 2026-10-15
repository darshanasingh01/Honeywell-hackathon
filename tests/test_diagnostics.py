"""Focused tests for the three-sensor envelope and fusion safeguards."""

from __future__ import annotations

import copy
import unittest

from ml.diagnostics.adapters.acoustic import wrap_acoustic_output
from ml.diagnostics.adapters.vibration import wrap_vibration_output
from ml.diagnostics.fusion import fuse_predictions
from ml.diagnostics.schema import SchemaValidationError, validate_prediction


def result(sensor: str, *, score: float = 1.0, threshold: float = 0.5,
           alert: bool = True, engine: str | None = "engine-A",
           session: str | None = "session-1", event: str | None = "event-1",
           start: str | None = "2026-01-01T00:00:00Z",
           end: str | None = "2026-01-01T00:00:10Z",
           status: str = "available", quality: str = "good") -> dict:
    return {
        "schema_version": "1.0", "prediction_id": f"prediction-{sensor}",
        "event_id": event, "sensor_type": sensor, "status": status,
        "engine_id": engine, "session_id": session,
        "measurement_interval": {"start_utc": start, "end_utc": end},
        "rpm": None,
        "anomaly": ({"score": score, "threshold": threshold,
                     "is_unusual": alert, "score_name": f"{sensor}_score"}
                    if status == "available" else None),
        "fault_prediction": None, "fault_probabilities": None,
        "signal_quality": {"status": quality, "metrics": {}},
        "model_version": "test-v1" if status == "available" else None,
        "limitations": [],
        "error": ({"code": "test_error", "message": "sensor failed"} if status == "error" else None),
        "native_output": None,
    }


class DiagnosticTests(unittest.TestCase):
    def test_acoustic_only_input_and_adapter_preserves_native_output(self):
        native = {
            "predicted_rpm": 1500, "rpm_confidence": None,
            "acoustic_novelty_score": 0.12, "novelty_threshold": 0.1,
            "is_acoustically_unusual": True, "fault_prediction": None,
            "model_version": "rpm-v1+novelty-v1", "limitations": ["unlabeled baseline"],
        }
        adapted = wrap_acoustic_output(native, prediction_id="p1")
        self.assertEqual(adapted["native_output"], native)
        self.assertEqual(adapted["rpm"]["value"], 1500)
        self.assertIsNone(adapted["fault_prediction"])
        fused = fuse_predictions([adapted])
        self.assertEqual(fused["fusion_status"], "single_sensor")
        self.assertEqual(fused["fusion_decision"], "review_recommended")
        self.assertEqual(fused["sensor_results"], [adapted])

    def test_matching_ids_and_overlapping_intervals_fuse(self):
        fused = fuse_predictions([result("acoustic"), result("vibration", alert=False)])
        self.assertEqual(fused["fusion_status"], "fused")
        self.assertEqual(fused["fusion_decision"], "review_recommended")
        self.assertIsNone(fused["fault_prediction"])

    def test_conflicting_engine_ids_do_not_fuse(self):
        fused = fuse_predictions([result("acoustic"), result("vibration", engine="engine-B")])
        self.assertEqual(fused["fusion_status"], "not_fused")
        self.assertEqual(fused["fusion_decision"], "review_recommended")

    def test_missing_event_metadata_does_not_fuse(self):
        fused = fuse_predictions([result("acoustic"), result("vibration", event=None)])
        self.assertEqual(fused["fusion_status"], "not_fused")

    def test_nonoverlapping_intervals_do_not_fuse(self):
        fused = fuse_predictions([
            result("acoustic", start="2026-01-01T00:00:00Z", end="2026-01-01T00:00:05Z"),
            result("vibration", start="2026-01-01T00:00:05Z", end="2026-01-01T00:00:10Z"),
        ])
        self.assertEqual(fused["fusion_status"], "not_fused")

    def test_unavailable_sensor_is_no_evidence(self):
        unavailable = {
            "schema_version": "1.0", "prediction_id": "p-v",
            "event_id": "event-1", "sensor_type": "vibration", "status": "unavailable",
            "engine_id": None, "session_id": None,
            "measurement_interval": {"start_utc": None, "end_utc": None},
            "rpm": None, "anomaly": None, "fault_prediction": None,
            "fault_probabilities": None,
            "signal_quality": {"status": "unavailable", "metrics": {}},
            "model_version": None, "limitations": ["No trained vibration model."],
            "error": None, "native_output": None,
        }
        fused = fuse_predictions([result("acoustic"), unavailable])
        self.assertEqual(fused["fusion_status"], "not_fused")
        self.assertEqual([e["sensor_type"] for e in fused["anomaly_evidence"]], ["acoustic"])

    def test_poor_quality_sensor_is_no_evidence(self):
        fused = fuse_predictions([result("acoustic"), result("vibration", quality="poor")])
        self.assertEqual(fused["fusion_status"], "not_fused")
        self.assertEqual([e["sensor_type"] for e in fused["anomaly_evidence"]], ["acoustic"])

    def test_poor_quality_vibration_adapter_does_not_corroborate(self):
        native = {
            "predicted_bearing_condition": "faulty_bearing",
            "class_probabilities_uncalibrated": {"healthy_bearing": 0.2, "faulty_bearing": 0.8},
            "model_version": "nust-v1", "input_channel": "Channel 3",
            "signal_quality": {"status": "poor", "metrics": {"sample_rate_verified": False}},
            "limitations": [
                "Probabilities are model scores and are not calibrated probabilities.",
                "The classifier predicts only the dataset's bearing-condition labels; it is not a combustion-knock classifier.",
            ],
        }
        vibration = wrap_vibration_output(native, prediction_id="v1")
        messages = vibration["limitations"]
        self.assertEqual(sum("uncalibrated model scores, not verified fault probabilities" in text for text in messages), 1)
        self.assertEqual(sum("bearing-condition labels" in text for text in messages), 1)
        fused = fuse_predictions([result("acoustic"), vibration])
        self.assertEqual(fused["fusion_status"], "not_fused")
        self.assertEqual(fused["fusion_decision"], "review_recommended")
        self.assertEqual([entry["sensor_type"] for entry in fused["anomaly_evidence"]], ["acoustic"])

    def test_anomaly_score_scales_are_retained_separately(self):
        fused = fuse_predictions([
            result("acoustic", score=0.08, threshold=0.07),
            result("vibration", score=1200.0, threshold=1000.0),
        ])
        scores = {e["sensor_type"]: e["score"] for e in fused["anomaly_evidence"]}
        self.assertEqual(scores, {"acoustic": 0.08, "vibration": 1200.0})
        self.assertNotIn("combined_score", fused)
        self.assertEqual(fused["fusion_decision"], "multi_sensor_corroboration")

    def test_duplicate_sensor_results_never_create_independent_corroboration(self):
        duplicate_a = result("acoustic")
        duplicate_b = copy.deepcopy(duplicate_a)
        duplicate_b["prediction_id"] = "another-request-id"
        fused = fuse_predictions([duplicate_a, duplicate_b, result("vibration", alert=False)])
        self.assertEqual(fused["fusion_status"], "not_fused")
        self.assertNotEqual(fused["fusion_decision"], "multi_sensor_corroboration")
        self.assertEqual(len(fused["sensor_results"]), 3)

    def test_malformed_schema_input_is_rejected(self):
        malformed = result("acoustic")
        malformed["sensor_type"] = "temperature"
        with self.assertRaises(SchemaValidationError):
            validate_prediction(malformed)
        with self.assertRaises(SchemaValidationError):
            fuse_predictions([malformed])


if __name__ == "__main__":
    unittest.main()
