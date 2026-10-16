"""RPM estimator, missing-tooth accounting, and CKP diagnostics adapter tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ml.ckp.analyze import analyze_csv, estimate_rpm
from ml.diagnostics.adapters.ckp import wrap_ckp_output
from ml.diagnostics.schema import validate_prediction


DATASET = Path("ml/datasets/ckp/synthetic")


class CkpRpmAnalysisTests(unittest.TestCase):
    def test_constant_reference_cases_match_rpm_ground_truth(self):
        for rpm in (800, 1500, 3000):
            with self.subTest(rpm=rpm):
                output = analyze_csv(DATASET / f"constant_{rpm}rpm_reference.csv")
                self.assertEqual(output["status"], "available", output.get("reason"))
                # Sampled pulse edges quantize the ten-tooth timing window;
                # allow a small ±5 RPM discretization envelope at 3000 RPM.
                self.assertAlmostEqual(output["estimated_rpm"], rpm, delta=5.0)
                comparison = output["ground_truth_comparison"]
                self.assertLess(comparison["mean_absolute_percentage_error"], 0.5)
                self.assertTrue(output["metrics"]["missing_gap_sequence_valid"])

    def test_variable_reference_case_tracks_rpm_profile(self):
        output = analyze_csv(DATASET / "variable_rpm_reference.csv")
        self.assertEqual(output["status"], "available", output.get("reason"))
        self.assertLess(output["ground_truth_comparison"]["mean_absolute_percentage_error"], 1.0)
        self.assertGreater(output["ground_truth_comparison"]["compared_intervals"], 1000)

    def test_missing_tooth_gap_is_accounted_as_three_tooth_pitches(self):
        output = analyze_csv(DATASET / "constant_1500rpm_reference.csv")
        self.assertEqual(output["status"], "available")
        self.assertEqual(output["metrics"]["missing_teeth"], 2)
        self.assertEqual(output["metrics"]["missing_gap_nominal_angle_deg"], 18.0)
        self.assertEqual(
            output["metrics"]["missing_gap_interval_spacing_pulses"],
            [58] * (output["missing_gap_interval_count"] - 1),
        )

    def test_4000_hz_stress_is_unavailable_below_two_samples_per_tooth(self):
        output = analyze_csv(DATASET / "constant_3000rpm_4000hz_stress.csv")
        self.assertEqual(output["status"], "unavailable")
        self.assertIn(output["reason"], {
            "sampling_resolution_below_2_samples_per_tooth",
            "unexplained_long_or_short_inter_pulse_intervals_possible_missed_pulses",
            "normal_tooth_intervals_could_not_be_identified",
        })

    def test_generator_underresolved_case_is_unavailable(self):
        output = analyze_csv(DATASET / "constant_3000rpm_1000hz_underresolved.csv")
        self.assertEqual(output["status"], "unavailable")

    def test_unexpected_missing_pulse_is_unavailable(self):
        source_path = DATASET / "constant_800rpm_reference.csv"
        with source_path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames
            rows = list(reader)
        removed = False
        for row in rows:
            if row["ckp_pulse"] == "1" and float(row["time_s"]) > 0.2:
                row["ckp_pulse"] = "0"
                removed = True
                break
        self.assertTrue(removed)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / source_path.name
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            path.with_suffix(".metadata.json").write_text(
                source_path.with_suffix(".metadata.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            result = analyze_csv(path)
        self.assertEqual(result["status"], "unavailable")

    def test_invalid_timestamps_are_unavailable(self):
        pulse = np.asarray([1, 0, 0, 1, 0, 1, 0, 0], dtype=np.uint8)
        duplicate_time = estimate_rpm(
            [0.0, 0.00001, 0.00002, 0.00002, 0.00004, 0.00005, 0.00006, 0.00007],
            pulse, signal_hz=100_000,
        )
        self.assertEqual(duplicate_time["status"], "unavailable")
        nan_time = estimate_rpm(
            [0.0, 0.00001, np.nan, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007],
            pulse, signal_hz=100_000,
        )
        self.assertEqual(nan_time["status"], "unavailable")

    def test_adapter_uses_shared_schema_and_never_claims_fault(self):
        native = analyze_csv(DATASET / "constant_800rpm_reference.csv")
        envelope = wrap_ckp_output(native, prediction_id="ckp-test")
        validate_prediction(envelope)
        self.assertEqual(envelope["sensor_type"], "ckp")
        self.assertEqual(envelope["rpm"]["value"], native["estimated_rpm"])
        self.assertIsNone(envelope["fault_prediction"])
        self.assertIsNone(envelope["fault_probabilities"])
        self.assertIsNone(envelope["engine_id"])
        self.assertIn("synthetic", " ".join(envelope["limitations"]).lower())
        self.assertNotIn("ground_truth_comparison", envelope["native_output"])

    def test_analysis_reads_metadata_rate_and_invalid_shape_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "short.csv"
            path.write_text("time_s,ckp_pulse\n0,0\n0.001,0\n", encoding="utf-8")
            sidecar = path.with_suffix(".metadata.json")
            sidecar.write_text(json.dumps({"sampling": {"sample_rate_hz": 1000}}), encoding="utf-8")
            result = analyze_csv(path)
            self.assertEqual(result["status"], "unavailable")
            self.assertIn("at_least_4_samples", result["reason"])


if __name__ == "__main__":
    unittest.main()
