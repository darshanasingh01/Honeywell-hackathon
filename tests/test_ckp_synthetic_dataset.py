"""Tests for synthetic CKP dataset generation and pulse validation."""

from __future__ import annotations

import unittest

import numpy as np

from ml.ckp.generate_synthetic_dataset import (
    DEFAULT_SEED,
    NOMINAL_TEETH,
    MISSING_TEETH,
    SIMULATOR_COMMIT,
    case_specs,
    load_pinned_generator,
    make_case,
)


class SyntheticCkpDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate_ckp, cls.source_sha256 = load_pinned_generator()
        cls.generate_ckp = staticmethod(generate_ckp)
    def spec(self, case_id):
        return next(item for item in case_specs() if item["case_id"] == case_id)

    def test_constant_reference_cases_have_exact_truth_and_valid_pulse_structure(self):
        for rpm in (800, 1500, 3000):
            with self.subTest(rpm=rpm):
                case = make_case(self.spec(f"constant_{rpm}rpm_reference"), self.generate_ckp)
                validation = case["validation"]
                self.assertEqual(case["metadata"]["ground_truth"]["rpm_constant"], float(rpm))
                self.assertTrue(validation["pulse_count_matches_ideal"])
                self.assertEqual(validation["missed_pulse_count"], 0)
                self.assertEqual(validation["case_status"], "usable_high_rate_synthetic_reference")
                self.assertTrue(validation["missing_gap_detection_reliable"])
                self.assertEqual(
                    validation["observed_missing_tooth_gap_count"],
                    validation["expected_missing_tooth_gap_count"],
                )

    def test_variable_rpm_profile_is_integrated_and_tagged_synthetic(self):
        case = make_case(self.spec("variable_rpm_reference"), self.generate_ckp)
        rpm = case["rpm_ground_truth"]
        self.assertGreater(float(np.max(rpm)), float(np.min(rpm)))
        self.assertEqual(case["metadata"]["provenance"].split(";")[0], "SYNTHETIC")
        self.assertTrue(case["validation"]["pulse_count_matches_ideal"])
        self.assertEqual(case["validation"]["case_status"], "usable_high_rate_synthetic_reference")

    def test_4000_hz_3000_rpm_is_reported_as_resolution_limited(self):
        case = make_case(self.spec("constant_3000rpm_4000hz_stress"), self.generate_ckp)
        validation = case["validation"]
        self.assertEqual(case["metadata"]["sampling"]["sample_rate_hz"], 4000)
        self.assertAlmostEqual(validation["samples_per_nominal_tooth_at_max_rpm"], 4 / 3, places=2)
        self.assertEqual(validation["case_status"], "stress_only_resolution_limited")

    def test_severe_under_sampling_reports_missed_pulses(self):
        case = make_case(self.spec("constant_3000rpm_1000hz_underresolved"), self.generate_ckp)
        self.assertFalse(case["validation"]["pulse_count_matches_ideal"])
        self.assertGreater(case["validation"]["missed_pulse_count"], 0)
        self.assertEqual(case["validation"]["case_status"], "unusable_underresolved_pulse_output")

    def test_repeat_runs_are_bitwise_reproducible(self):
        spec = self.spec("constant_1500rpm_reference")
        first = make_case(spec, self.generate_ckp, seed=DEFAULT_SEED)
        second = make_case(spec, self.generate_ckp, seed=DEFAULT_SEED)
        for name in ("time_s", "crank_angle_deg", "rpm_ground_truth", "ckp_pulse"):
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(self.source_sha256, load_pinned_generator()[1])
        self.assertEqual(SIMULATOR_COMMIT, "92e69cf6b87c1417c87a61998d391598e53d7081")
        self.assertEqual(NOMINAL_TEETH, 60)
        self.assertEqual(MISSING_TEETH, 2)


if __name__ == "__main__":
    unittest.main()
