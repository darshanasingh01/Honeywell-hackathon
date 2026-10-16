"""Estimate RPM from a sampled CKP pulse train using missing-tooth geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODEL_VERSION = "engineguard-ckp-rpm-60-2-v1"
DEFAULT_TEETH = 60
DEFAULT_MISSING_TEETH = 2
_MIN_SAMPLES_PER_TOOTH = 2.0
_GOOD_SAMPLES_PER_TOOTH = 10.0
_RPM_WINDOW_TOOTH_INTERVALS = 10


def _unavailable(reason: str, **metrics) -> dict:
    return {
        "status": "unavailable",
        "estimated_rpm": None,
        "mean_estimated_rpm": None,
        "quality_status": "unavailable",
        "reason": reason,
        "metrics": metrics,
        "missing_gap_interval_count": None,
        "interval_count": 0,
        "model_version": MODEL_VERSION,
        "limitations": [
            "RPM estimation is unavailable because the CKP input failed validation or lacked sufficient sampling resolution.",
            "This algorithm estimates rotational speed only; it is not a fault detector.",
        ],
    }


def _edge_indices(pulse: np.ndarray) -> np.ndarray:
    previous_low = np.r_[True, pulse[:-1] == 0]
    return np.flatnonzero((pulse == 1) & previous_low)


def _local_normal_periods(intervals: np.ndarray) -> np.ndarray:
    """Estimate each normal tooth interval from neighboring, non-current periods."""
    baselines = np.empty(intervals.size, dtype=np.float64)
    for index in range(intervals.size):
        left = intervals[max(0, index - 8):index]
        right = intervals[index + 1:min(intervals.size, index + 9)]
        neighbors = np.concatenate((left, right))
        if neighbors.size < 3:
            baselines[index] = np.nan
        else:
            baselines[index] = np.median(neighbors)
    return baselines


def estimate_rpm(
    time_s,
    pulse,
    *,
    signal_hz: int,
    nominal_teeth: int = DEFAULT_TEETH,
    missing_teeth: int = DEFAULT_MISSING_TEETH,
) -> dict:
    """Estimate RPM per pulse interval; account for the 60-2 triple-length gap.

    Input pulse samples are binary. Invalid timestamps, unexplained pulse gaps,
    or fewer than two samples per tooth produce an unavailable result. Rates
    below ten samples/tooth remain available but are marked poor quality.
    """
    if isinstance(signal_hz, bool) or not isinstance(signal_hz, (int, np.integer)) or signal_hz <= 0:
        return _unavailable("invalid_sample_rate_hz")
    if (isinstance(nominal_teeth, bool) or not isinstance(nominal_teeth, (int, np.integer))
            or isinstance(missing_teeth, bool) or not isinstance(missing_teeth, (int, np.integer))
            or nominal_teeth < 4 or not 1 <= missing_teeth < nominal_teeth - 1):
        return _unavailable("invalid_tooth_geometry")

    try:
        times = np.asarray(time_s, dtype=np.float64)
        raw_pulse = np.asarray(pulse, dtype=np.float64)
    except (TypeError, ValueError):
        return _unavailable("timestamps_or_pulses_are_not_numeric")
    if times.ndim != 1 or raw_pulse.ndim != 1 or times.size != raw_pulse.size or times.size < 4:
        return _unavailable("timestamps_and_pulses_must_be_aligned_1d_arrays_with_at_least_4_samples")
    if not np.isfinite(times).all() or not np.isfinite(raw_pulse).all():
        return _unavailable("timestamps_or_pulses_contain_nonfinite_values")
    if np.any((raw_pulse != 0) & (raw_pulse != 1)):
        return _unavailable("pulse_signal_must_be_binary")
    if np.any(np.diff(times) <= 0):
        return _unavailable("timestamps_must_be_strictly_increasing")

    dt = np.diff(times)
    nominal_dt = 1.0 / signal_hz
    if not np.allclose(dt, nominal_dt, rtol=1e-3, atol=nominal_dt * 1e-4):
        return _unavailable(
            "timestamps_do_not_match_declared_uniform_sample_rate",
            declared_sample_rate_hz=int(signal_hz),
            median_sample_interval_s=float(np.median(dt)),
        )

    pulse_bits = raw_pulse.astype(np.uint8)
    edge_indices = _edge_indices(pulse_bits)
    if edge_indices.size < 5:
        return _unavailable("too_few_pulse_edges", pulse_edge_count=int(edge_indices.size))
    edge_times = times[edge_indices]
    intervals = np.diff(edge_times)
    local_period = _local_normal_periods(intervals)
    if not np.isfinite(local_period).all() or np.any(local_period <= 0):
        return _unavailable("too_few_neighboring_pulse_intervals", pulse_edge_count=int(edge_indices.size))

    pitch_deg = 360.0 / int(nominal_teeth)
    missing_gap_factor = int(missing_teeth) + 1
    ratio = intervals / local_period
    normal = (ratio >= 0.60) & (ratio <= 1.65)
    missing_gap = (ratio >= 2.10) & (ratio <= 4.10)
    unclassified = ~(normal | missing_gap)
    normal_intervals = intervals[normal]
    if normal_intervals.size < 3:
        return _unavailable("normal_tooth_intervals_could_not_be_identified", pulse_edge_count=int(edge_indices.size))

    samples_per_tooth = float(np.min(normal_intervals) * signal_hz)
    quality_metrics = {
        "sample_rate_hz": int(signal_hz),
        "nominal_teeth_per_revolution": int(nominal_teeth),
        "missing_teeth": int(missing_teeth),
        "pulse_edge_count": int(edge_indices.size),
        "inter_pulse_interval_count": int(intervals.size),
        "missing_gap_interval_count": int(np.sum(missing_gap)),
        "unclassified_interval_count": int(np.sum(unclassified)),
        "minimum_observed_samples_per_tooth": samples_per_tooth,
        "good_quality_minimum_samples_per_tooth": _GOOD_SAMPLES_PER_TOOTH,
        "minimum_usable_samples_per_tooth": _MIN_SAMPLES_PER_TOOTH,
    }
    if samples_per_tooth < _MIN_SAMPLES_PER_TOOTH:
        return _unavailable("sampling_resolution_below_2_samples_per_tooth", **quality_metrics)
    if np.any(unclassified):
        return _unavailable("unexplained_long_or_short_inter_pulse_intervals_possible_missed_pulses", **quality_metrics)

    gap_indices = np.flatnonzero(missing_gap)
    gap_spacing = np.diff(gap_indices)
    gap_sequence_valid = bool(gap_spacing.size == 0 or np.all(gap_spacing == nominal_teeth - missing_teeth))
    expected_gaps_approx = int(round(intervals.size / (nominal_teeth - missing_teeth)))
    gap_count_consistent = abs(int(gap_indices.size) - expected_gaps_approx) <= 1
    quality_metrics.update({
        "missing_gap_interval_indices": gap_indices.astype(int).tolist(),
        "missing_gap_interval_spacing_pulses": gap_spacing.astype(int).tolist(),
        "expected_missing_gap_interval_spacing_pulses": int(nominal_teeth - missing_teeth),
        "missing_gap_sequence_valid": gap_sequence_valid,
        "missing_gap_count_consistent_with_record_length": bool(gap_count_consistent),
        "missing_gap_nominal_angle_deg": float(pitch_deg * missing_gap_factor),
    })
    if gap_indices.size == 0 or not gap_sequence_valid or not gap_count_consistent:
        return _unavailable("60_minus_2_missing_tooth_gap_pattern_not_verified", **quality_metrics)

    factors = np.where(missing_gap, float(missing_gap_factor), 1.0)
    # Pool ten tooth intervals to reduce quantization error from sampled edges.
    # The missing gap contributes three nominal intervals to angular travel.
    window_size = _RPM_WINDOW_TOOTH_INTERVALS
    if intervals.size < window_size:
        return _unavailable(
            "too_few_pulses_for_rpm_timing_window",
            **quality_metrics,
            required_tooth_intervals=window_size,
        )
    window_rpm = np.empty(intervals.size - window_size + 1, dtype=np.float64)
    window_midpoint_s = np.empty(window_rpm.size, dtype=np.float64)
    for start in range(window_rpm.size):
        stop = start + window_size
        elapsed = edge_times[stop] - edge_times[start]
        # Degrees = sum(factor * tooth pitch); RPM = degrees * 60 / (360 * time).
        window_rpm[start] = np.sum(factors[start:stop]) * 60.0 / nominal_teeth / elapsed
        window_midpoint_s[start] = (edge_times[start] + edge_times[stop]) / 2.0
    if not np.isfinite(window_rpm).all() or np.any(window_rpm <= 0):
        return _unavailable("rpm_estimates_are_not_finite_positive_values", **quality_metrics)
    quality_status = "good" if samples_per_tooth >= _GOOD_SAMPLES_PER_TOOTH else "poor"
    quality_metrics["rpm_estimation_window_tooth_intervals"] = int(window_size)
    quality_metrics["rpm_estimation_window_count"] = int(window_rpm.size)
    return {
        "status": "available",
        "estimated_rpm": float(np.median(window_rpm)),
        "mean_estimated_rpm": float(np.mean(window_rpm)),
        "quality_status": quality_status,
        "reason": None,
        "metrics": quality_metrics,
        "missing_gap_interval_count": int(gap_indices.size),
        "interval_count": int(intervals.size),
        "model_version": MODEL_VERSION,
        "limitations": [
            "RPM is estimated from edge timing under the declared missing-tooth wheel geometry.",
            "This algorithm estimates rotational speed only; it is not a fault detector.",
        ],
        "_window_rpm": window_rpm,
        "_window_midpoint_s": window_midpoint_s,
    }


def _ground_truth_comparison(analysis: dict, time_s: np.ndarray, rpm_truth: np.ndarray) -> dict | None:
    if analysis["status"] != "available":
        return None
    truth = np.interp(analysis["_window_midpoint_s"], time_s, rpm_truth)
    estimated = analysis["_window_rpm"]
    error = estimated - truth
    absolute_error = np.abs(error)
    percentage_error = absolute_error / truth * 100.0
    return {
        "comparison_basis": "ten-tooth-window estimate versus linearly interpolated synthetic RPM ground truth at window midpoint",
        "compared_intervals": int(estimated.size),
        "mean_absolute_error_rpm": float(np.mean(absolute_error)),
        "root_mean_squared_error_rpm": float(np.sqrt(np.mean(error ** 2))),
        "mean_absolute_percentage_error": float(np.mean(percentage_error)),
        "maximum_absolute_error_rpm": float(np.max(absolute_error)),
        "median_absolute_percentage_error": float(np.median(percentage_error)),
    }


def analyze_csv(
    csv_path: str | Path,
    *,
    sample_rate_hz: int | None = None,
    nominal_teeth: int | None = None,
    missing_teeth: int | None = None,
) -> dict:
    """Analyze one CKP CSV, using sidecar metadata when present."""
    path = Path(csv_path)
    sidecar_path = path.with_suffix(".metadata.json")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    sampling = metadata.get("sampling", {})
    geometry = metadata.get("tooth_geometry", {})
    fs = sample_rate_hz if sample_rate_hz is not None else sampling.get("sample_rate_hz")
    teeth = nominal_teeth if nominal_teeth is not None else geometry.get("nominal_teeth_per_revolution", DEFAULT_TEETH)
    missing = missing_teeth if missing_teeth is not None else geometry.get("missing_teeth", DEFAULT_MISSING_TEETH)
    if fs is None:
        result = _unavailable("sample_rate_hz_required_when_metadata_is_absent")
        result["input_file"] = str(path)
        return result

    required = {"time_s", "ckp_pulse"}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CKP CSV must contain time_s and ckp_pulse columns")
        rows = list(reader)
    try:
        times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
        pulse = np.asarray([float(row["ckp_pulse"]) for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return {**_unavailable("time_s_or_ckp_pulse_contains_non_numeric_data"), "input_file": str(path)}
    result = estimate_rpm(times, pulse, signal_hz=fs, nominal_teeth=teeth, missing_teeth=missing)
    result["input_file"] = str(path)
    result["sampling_rate_hz"] = int(fs)
    result["tooth_geometry"] = {"nominal_teeth": int(teeth), "missing_teeth": int(missing)}
    result["provenance"] = metadata.get("provenance")
    result["synthetic"] = bool(metadata.get("data_kind") == "synthetic_ckp_waveform")
    if result["synthetic"]:
        result["limitations"].append(
            "Synthetic CKP data validates software behavior only and does not establish real-world accuracy."
        )
    if result["status"] == "available" and rows and "rpm_ground_truth" in rows[0]:
        try:
            rpm_truth = np.asarray([float(row["rpm_ground_truth"]) for row in rows], dtype=np.float64)
        except (TypeError, ValueError):
            rpm_truth = np.asarray([], dtype=np.float64)
        if rpm_truth.size == times.size and np.isfinite(rpm_truth).all():
            result["ground_truth_comparison"] = _ground_truth_comparison(result, times, rpm_truth)
    result.pop("_window_rpm", None)
    result.pop("_window_midpoint_s", None)
    return result


def evaluate_reference_dataset(dataset_dir: str | Path) -> dict:
    """Evaluate manifest-declared high-rate reference CSVs against synthetic RPM truth."""
    root = Path(dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    for case in manifest["cases"]:
        if case.get("case_status") != "usable_high_rate_synthetic_reference":
            continue
        result = analyze_csv(root / case["csv"])
        results[case["case_id"]] = result
    return {
        "dataset_provenance": manifest.get("provenance"),
        "generator_source_commit": manifest.get("generator_source_commit"),
        "model_version": MODEL_VERSION,
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="One waveform CSV")
    parser.add_argument("--sample-rate-hz", type=int, help="Required if input has no metadata sidecar")
    parser.add_argument("--nominal-teeth", type=int, default=DEFAULT_TEETH)
    parser.add_argument("--missing-teeth", type=int, default=DEFAULT_MISSING_TEETH)
    parser.add_argument("--evaluate-dataset", action="store_true", help="Evaluate manifest-declared reference cases")
    parser.add_argument("--dataset-dir", type=Path, default=Path("ml/datasets/ckp/synthetic"))
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.evaluate_dataset:
        result = evaluate_reference_dataset(args.dataset_dir)
    elif args.input:
        result = analyze_csv(args.input, sample_rate_hz=args.sample_rate_hz,
                             nominal_teeth=args.nominal_teeth, missing_teeth=args.missing_teeth)
    else:
        parser.error("provide an input CSV or --evaluate-dataset")
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0 if not args.input or result.get("status") == "available" else 1


if __name__ == "__main__":
    raise SystemExit(main())
