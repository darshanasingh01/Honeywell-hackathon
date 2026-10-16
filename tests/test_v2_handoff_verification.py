import unittest

from ml.synthetic_conditions.verify_v2_handoff import verify


class V2HandoffVerificationTests(unittest.TestCase):
    def test_handoff_is_complete_and_hash_verified(self):
        result = verify()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["sensor_windows"], 5)
        self.assertEqual(result["spectrogram_files"], 5)
        self.assertEqual(len(result["models"]), 3)


if __name__ == "__main__":
    unittest.main()
