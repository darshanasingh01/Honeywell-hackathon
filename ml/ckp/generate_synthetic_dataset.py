"""Generate and validate synthetic CKP records with the pinned sim-branch model.

All outputs are explicitly synthetic. The CKP implementation is loaded from
the immutable Git commit recorded below; this script does not copy or modify
the teammate's simulator source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import types

import numpy as np


SIMULATOR_COMMIT = "92e69cf6b87c1417c87a61998d391598e53d7081"
SIMULATOR_PATH = "simulator/ckp_model.py"
NOMINAL_TEETH = 60
MISSING_TEETH = 2
REFERENCE_HZ = 100_000
DURATION_S = 1.0
DEFAULT_SEED = 42
OUTPUT_DIR = Path("ml/datasets/ckp/synthetic")

VARIABLE_RPM_KNOTS = ((0.0, 800.0), (0.30, 3000.0), (0.70, 1500.0), (1.0, 2400.0))


def load_pinned_generator(repo_root: Path | None = None):
    """Load the exact teammate implementation from its pinned Git blob."""
    root = repo_root or Path(__file__).resolve().parents[2]
    command = ["git", "show", f"{SIMULATOR_COMMIT}:{SIMULATOR_PATH}"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or f"git show exited {completed.returncode}"
        raise RuntimeError(
            f"Cannot load pinned CKP generator {SIMULATOR_COMMIT}:{SIMULATOR_PATH}: {detail}. "
            "Fetch origin/sim so the pinned commit is present, without switching branches."
        )
    source = completed.stdout
    module = types.ModuleType("engineguard_pinned_sim_ckp_model")
    module.__file__ = f"{SIMULATOR_COMMIT}:{SIMULATOR_PATH}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    generator = getattr(module, "generate_ckp", None)
    if not callable(generator):
        raise RuntimeError("Pinned simulator source does not expose callable generate_ckp")
    return generator, hashlib.sha256(source.encode("utf-8")).hexdigest()


def case_specs() -> list[dict]:
    specs = [
        {"case_id": f"constant_{rpm}rpm_reference", "rpm": float(rpm), "signal_hz": REFERENCE_HZ}
        for rpm in (800, 1500, 3000)
    ]
    specs.append({
        "case_id": "variable_rpm_reference", "rpm_profile": VARIABLE_RPM_KNOTS,
        "signal_hz": REFERENCE_HZ,
    })
    specs.extend(
        {"case_id": f"constant_{rpm}rpm_4000hz_stress", "rpm": float(rpm), "signal_hz": 4000}
        for rpm in (800, 1500, 3000)
    )
    # Diagnostic under-resolution case verifies that the validator detects lost tooth events.
    specs.append({"case_id": "constant_3000rpm_1000hz_underresolved", "rpm": 3000.0, "signal_hz": 1000})
    return specs


def make_case(spec: dict, generate_ckp, *, duration_s: float = DURATION_S, seed: int = DEFAULT_SEED) -> dict:
    """Construct angle from declared RPM truth and call the pinned CKP model."""
    signal_hz = int(spec["signal_hz"])
    if duration_s <= 0 or signal_hz <= 0:
        raise ValueError("duration_s and signal_hz must be positive")
    time_s = np.arange(0.0, duration_s, 1.0 / signal_hz, dtype=np.float64)
    if time_s.size < 2:
        raise ValueError("Case must contain at least two samples")

    if "rpm_profile" in spec:
        knots = np.asarray(spec["rpm_profile"], dtype=np.float64)
        if knots.ndim != 2 or knots.shape[1] != 2 or np.any(np.diff(knots[:, 0]) <= 0):
            raise ValueError("rpm_profile must contain ordered (time_s, rpm) pairs")
        rpm_truth = np.interp(time_s, knots[:, 0], knots[:, 1])
        profile_metadata = [[float(t), float(rpm)] for t, rpm in knots]
    else:
        rpm_truth = np.full(time_s.shape, float(spec["rpm"]), dtype=np.float64)
        profile_metadata = None
    if not np.isfinite(rpm_truth).all() or np.any(rpm_truth <= 0):
        raise ValueError("RPM truth must be finite and positive")

    # Trapezoidal discrete integration: d(angle)/dt = RPM * 6 degrees/second.
    crank_angle_deg = np.empty_like(time_s)
    crank_angle_deg[0] = 0.0
    crank_angle_deg[1:] = np.cumsum((rpm_truth[:-1] + rpm_truth[1:]) * (3.0 / signal_hz))
    pulse = np.asarray(
        generate_ckp(
            crank_angle_deg,
            signal_hz,
            nominal_teeth=NOMINAL_TEETH,
            missing_teeth=MISSING_TEETH,
        ),
        dtype=np.uint8,
    )
    if pulse.shape != time_s.shape or not np.isin(pulse, (0, 1)).all():
        raise ValueError("Pinned generate_ckp returned unexpected output shape or non-binary values")

    validation = validate_pulse_output(crank_angle_deg, pulse, rpm_truth, signal_hz)
    return {
        "time_s": time_s,
        "crank_angle_deg": crank_angle_deg,
        "rpm_ground_truth": rpm_truth,
        "ckp_pulse": pulse,
        "validation": validation,
        "metadata": {
            "case_id": spec["case_id"],
            "provenance": "SYNTHETIC; generated for EngineGuard AI analysis; not a real sensor measurement.",
            "data_kind": "synthetic_ckp_waveform",
            "ground_truth": {
                "rpm_constant": float(spec["rpm"]) if "rpm" in spec else None,
                "rpm_profile_knots_time_s_rpm": profile_metadata,
                "rpm_min": float(np.min(rpm_truth)),
                "rpm_max": float(np.max(rpm_truth)),
                "rpm_mean": float(np.mean(rpm_truth)),
            },
            "sampling": {
                "sample_rate_hz": signal_hz,
                "sample_interval_s": 1.0 / signal_hz,
                "sample_count": int(time_s.size),
                "duration_s_requested": float(duration_s),
                "duration_s_sampled": float(time_s[-1] - time_s[0]),
            },
            "tooth_geometry": {
                "nominal_teeth_per_revolution": NOMINAL_TEETH,
                "missing_teeth": MISSING_TEETH,
                "nominal_tooth_pitch_deg": 360.0 / NOMINAL_TEETH,
                "missing_gap_pitch_deg": (MISSING_TEETH + 1) * 360.0 / NOMINAL_TEETH,
            },
            "seed": int(seed),
            "seed_effect": "Recorded for run traceability; this deterministic CKP model adds no random noise.",
            "angle_integration": "Trapezoidal integration of declared RPM truth on the sampled time grid; initial angle 0 degrees.",
            "generator": {
                "source_commit": SIMULATOR_COMMIT,
                "source_path": SIMULATOR_PATH,
                "source_sha256": None,
                "function": "generate_ckp",
            },
            "pulse_validation": validation,
        },
    }


def validate_pulse_output(crank_angle_deg: np.ndarray, pulse: np.ndarray,
                          rpm_truth: np.ndarray, signal_hz: int) -> dict:
    """Check emitted pulses against ideal 60-2 tooth crossings and sampling limits."""
    angle = np.asarray(crank_angle_deg, dtype=np.float64)
    bits = np.asarray(pulse, dtype=np.uint8)
    rpm = np.asarray(rpm_truth, dtype=np.float64)
    pitch = 360.0 / NOMINAL_TEETH
    if angle.ndim != 1 or bits.shape != angle.shape or rpm.shape != angle.shape:
        raise ValueError("angle, pulse, and rpm arrays must be aligned 1-D arrays")
    if not np.isfinite(angle).all() or not np.isfinite(rpm).all() or np.any(np.diff(angle) <= 0):
        raise ValueError("angle and RPM truth arrays must be finite; crank angle strictly increasing")

    last_tooth = int(np.floor(angle[-1] / pitch + 1e-10))
    full_revolutions, final_slot = divmod(last_tooth, NOMINAL_TEETH)
    ideal_count = full_revolutions * (NOMINAL_TEETH - MISSING_TEETH)
    ideal_count += min(final_slot + 1, NOMINAL_TEETH - MISSING_TEETH)
    pulse_indices = np.flatnonzero(bits)
    pulse_angles = angle[pulse_indices]
    gaps_deg = np.diff(pulse_angles)
    max_angle_step = float(np.max(np.diff(angle)))
    samples_per_tooth_at_max_rpm = float(signal_hz / np.max(rpm))
    missing_gap_angle = (MISSING_TEETH + 1) * pitch
    gap_detection_reliable = max_angle_step <= pitch
    detected_missing_gaps = (
        int(np.sum(gaps_deg > 2.0 * pitch)) if gap_detection_reliable else None
    )
    expected_missing_gaps = full_revolutions
    missed_pulses = max(0, ideal_count - int(pulse_indices.size))
    pulse_count_matches_ideal = int(pulse_indices.size) == ideal_count
    gap_count_matches = (
        detected_missing_gaps == expected_missing_gaps
        if detected_missing_gaps is not None else None
    )
    high_resolution = samples_per_tooth_at_max_rpm >= 10.0
    usable_reference = bool(pulse_count_matches_ideal and gap_count_matches and high_resolution)
    if usable_reference:
        case_status = "usable_high_rate_synthetic_reference"
    elif not pulse_count_matches_ideal or max_angle_step > pitch:
        case_status = "unusable_underresolved_pulse_output"
    else:
        case_status = "stress_only_resolution_limited"

    return {
        "case_status": case_status,
        "ideal_pulse_count": int(ideal_count),
        "observed_pulse_count": int(pulse_indices.size),
        "missed_pulse_count": int(missed_pulses),
        "pulse_count_matches_ideal": bool(pulse_count_matches_ideal),
        "expected_missing_tooth_gap_count": int(expected_missing_gaps),
        "observed_missing_tooth_gap_count": detected_missing_gaps,
        "missing_tooth_gap_count_matches": gap_count_matches,
        "missing_tooth_gap_expected_angle_deg": missing_gap_angle,
        "max_sample_crank_angle_step_deg": max_angle_step,
        "samples_per_nominal_tooth_at_max_rpm": samples_per_tooth_at_max_rpm,
        "missing_gap_detection_reliable": bool(gap_detection_reliable),
        "high_resolution_reference_criterion_samples_per_tooth": 10.0,
        "resolution_note": (
            "At fewer than 10 samples per tooth, pulse timing is coarse; this is a stress case, not a reference."
            if not high_resolution else
            "At least 10 samples per tooth at the maximum RPM; this only validates the synthetic model output."
        ),
    }


def write_case(case: dict, output_dir: Path, source_sha256: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_id = case["metadata"]["case_id"]
    csv_path = output_dir / f"{case_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "crank_angle_deg", "rpm_ground_truth", "ckp_pulse"))
        writer.writerows(zip(
            (f"{x:.10f}" for x in case["time_s"]),
            (f"{x:.10f}" for x in case["crank_angle_deg"]),
            (f"{x:.8f}" for x in case["rpm_ground_truth"]),
            (str(int(x)) for x in case["ckp_pulse"]),
        ))
    metadata = case["metadata"]
    metadata["generator"]["source_sha256"] = source_sha256
    metadata_path = output_dir / f"{case_id}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return csv_path


def generate_dataset(output_dir: Path = OUTPUT_DIR, *, seed: int = DEFAULT_SEED) -> dict:
    generate_ckp, source_sha256 = load_pinned_generator()
    summaries = []
    for spec in case_specs():
        case = make_case(spec, generate_ckp, seed=seed)
        case["metadata"]["generator"]["source_sha256"] = source_sha256
        csv_path = write_case(case, output_dir, source_sha256)
        summaries.append({
            "case_id": case["metadata"]["case_id"],
            "csv": csv_path.name,
            "metadata": f"{case['metadata']['case_id']}.metadata.json",
            **case["validation"],
        })
    manifest = {
        "dataset_name": "EngineGuard AI synthetic CKP analysis dataset",
        "provenance": "SYNTHETIC ONLY; not a real engine, vehicle, or sensor recording.",
        "generator_source_commit": SIMULATOR_COMMIT,
        "generator_source_path": SIMULATOR_PATH,
        "generator_source_sha256": source_sha256,
        "tooth_geometry": {"nominal_teeth": NOMINAL_TEETH, "missing_teeth": MISSING_TEETH},
        "seed": int(seed),
        "cases": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = generate_dataset(args.output_dir, seed=args.seed)
    print(json.dumps({"output_dir": str(args.output_dir), "cases": manifest["cases"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
