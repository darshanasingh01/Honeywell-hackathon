"""Pinned-source integration checks; generated fixtures are synthetic-only."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.synthetic_conditions.ckp_v2 import process_edges
from ml.synthetic_conditions.producer_v2 import (
    CONDITIONS, FS, TICK, approved_sources, decoded_telemetry, generate_dataset,
    generate_run, map_pressure_kpa, producer_paths, sha256, tooth_events, validate_dataset,
)
from ml.synthetic_conditions.schema_v2 import FEATURE_COLUMNS, MANIFEST
from ml.synthetic_conditions.verify_producer_v2 import verify_directory


class ProducerV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.audit = generate_run("normal", 4100, 1.0, "integration")

    def test_all_features_covered_and_reproducible(self):
        again, audit = generate_run("normal", 4100, 1.0, "integration")
        self.assertEqual(self.rows, again)
        self.assertEqual(self.audit, audit)
        self.assertEqual(tuple(producer_paths()), FEATURE_COLUMNS)
        self.assertEqual(len(FEATURE_COLUMNS), 167)
        self.assertEqual(len(self.rows), 3)
        self.assertEqual([r["t_start"] for r in self.rows], [0, 0.25, 0.5])
        self.assertTrue(all(r["t_end"] - r["t_start"] == 0.5 for r in self.rows))

    def test_waveform_dsp_reuses_approved_values(self):
        simulator, dsp, _, hashes = approved_sources()
        _, wave = simulator.simulate_engine("normal", 1, 10, FS, 4100)
        for channel, prefix in (("microphone_audio", "mic"), ("knock_sensor", "knock"),
                                ("accel_x", "accel_x"), ("accel_y", "accel_y"), ("accel_z", "accel_z")):
            expected = dsp.extract_fast_channel_features(wave[channel].to_numpy()[:FS // 2], FS, prefix)
            for key, value in expected.items():
                self.assertEqual(self.rows[0][key], value)
        self.assertEqual(len(hashes), 3)
        self.assertTrue(all(len(h) == 64 for h in hashes.values()))

    def test_map_operating_direction_and_byte_consistency(self):
        self.assertGreater(map_pressure_kpa(0.8, 80, 1800), map_pressure_kpa(0.2, 20, 1800))
        self.assertGreater(map_pressure_kpa(0.5, 40, 1000), map_pressure_kpa(0.5, 40, 6000))
        sim = approved_sources()[0]
        source, _ = sim.simulate_engine("misfire", 1, 10, FS, 4101)
        decoded, payloads = decoded_telemetry(source)
        np.testing.assert_allclose(decoded.timestamp_s, source.timestamp_s, atol=0)
        np.testing.assert_allclose(decoded.engine_load_pct, decoded.engine_load * 100, atol=1e-12)
        self.assertLessEqual(np.max(np.abs(decoded.map_kpa - map_pressure_kpa(
            source.engine_load, source.throttle_pct, source.engine_rpm))), 0.5)
        self.assertEqual(decoded.map_kpa.tolist(), [float(p[0]) for p in payloads["0x0B"]])
        for suffix in ("mean", "std", "min", "max", "slope"):
            self.assertAlmostEqual(self.rows[0]["tel_engine_load_pct_" + suffix],
                                   self.rows[0]["tel_engine_load_" + suffix] * 100)

    def test_tooth_geometry_shared_phase_and_acquisition_failures(self):
        sim = approved_sources()[0]
        source, wave = sim.simulate_engine("misfire", 1, 10, FS, 4102)
        events = tooth_events(source, 1)
        times = np.array([t for t, _ in events])
        indices = np.array([i for _, i in events])
        self.assertTrue(np.all(np.diff(times) > 0))
        np.testing.assert_allclose(times / TICK, np.round(times / TICK), atol=1e-9)
        transition = (indices[:-1] == 57) & (indices[1:] == 0)
        self.assertGreater(transition.sum(), 10)
        self.assertTrue(np.all((indices[1:] == indices[:-1] + 1) | transition))
        rpm = np.interp(wave.timestamp_s, source.timestamp_s, source.engine_rpm)
        phase = np.cumsum(rpm / FS)
        crossing_phase = np.interp(times, wave.timestamp_s, phase)
        self.assertLess(np.max(np.abs(crossing_phase - np.round(crossing_phase))), 0.02)
        healthy = process_edges(events, start=0, end=1, timestamp_resolution_s=TICK)
        self.assertEqual(healthy["quality_status"], "good")
        missing = events[:30] + events[31:]
        for sequence, tick in ((missing, TICK), (events, 0.001)):
            damaged = process_edges(sequence, start=0, end=1, timestamp_resolution_s=tick)
            self.assertEqual(damaged["quality_status"], "poor_quality")
            self.assertIsNone(damaged["features"]["ckp_tooth_rpm_mean"])
        self.assertEqual(self.audit["condition"], "normal")  # quality is not a condition label

    def test_smoke_serialization_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "smoke"
            report = generate_dataset(output, seeds=(4103,), duration_s=0.5, namespace="test_smoke")
            self.assertEqual(report["rows"], 5)
            self.assertEqual(report["run_count"], 5)
            self.assertEqual(set(report["class_balance"]), set(CONDITIONS))
            frame = pd.read_csv(output / "features.csv", float_precision="round_trip")
            self.assertEqual(list(frame), MANIFEST["metadata_columns"] + list(FEATURE_COLUMNS))
            self.assertEqual(validate_dataset(frame)["status"], "PASS")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["real_engine_measurements"])
            self.assertEqual(manifest["features_sha256"], sha256((output / "features.csv").read_bytes()))
            verification = verify_directory(output)
            self.assertTrue(verification["source_hashes_and_saved_evidence_verified"])
            original = (output / "features.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                generate_dataset(output, seeds=(4103,), duration_s=0.5, namespace="test_smoke")
            self.assertEqual((output / "features.csv").read_bytes(), original)
            for mutation in (lambda f: f.assign(ckp_unexpected_gap_count=1.5),
                             lambda f: f.assign(tel_map_kpa_mean=np.nan),
                             lambda f: f.iloc[:, ::-1],
                             lambda f: pd.concat([f, f.iloc[:1]], ignore_index=True)):
                with self.assertRaises(ValueError):
                    validate_dataset(mutation(frame))
            audit_path = output / "run_evidence.json"
            audit = json.loads(audit_path.read_text())
            audit[0]["ckp_events"][0][0] += 0.000003
            audit_path.write_text(json.dumps(audit))
            with self.assertRaises(ValueError):
                verify_directory(output)


if __name__ == "__main__":
    unittest.main()
