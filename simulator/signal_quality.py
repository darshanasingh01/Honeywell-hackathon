"""Signal-quality checks for synthetic engine waveforms.

These checks flag data-quality risks; they do not diagnose engine faults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inspect_signal(
    values: np.ndarray | pd.Series,
    sample_hz: float,
    *,
    clipping_limit: float | None = None,
    expected_sample_hz: float | None = None,
) -> dict:
    """Return quality metrics and warnings for a sampled signal."""
    x = np.asarray(values, dtype=float).reshape(-1)
    warnings: list[str] = []

    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    if x.size < 2:
        raise ValueError("Signal must contain at least two samples")

    finite = np.isfinite(x)
    valid = x[finite]

    if not finite.all():
        warnings.append("non_finite_values")
    if valid.size < 2:
        raise ValueError("At least two finite samples are required")

    if np.std(valid) == 0:
        warnings.append("constant_signal")

    clipping_fraction = None
    if clipping_limit is not None:
        if clipping_limit <= 0:
            raise ValueError("clipping_limit must be positive")
        clipping_fraction = float(
            np.mean(np.abs(valid) >= clipping_limit)
        )
        if clipping_fraction > 0:
            warnings.append("possible_clipping")

    if expected_sample_hz is not None:
        if expected_sample_hz <= 0:
            raise ValueError("expected_sample_hz must be positive")
        if not np.isclose(sample_hz, expected_sample_hz):
            warnings.append("unexpected_sample_rate")

    # Nyquist is a hard limit: content at or above it cannot be
    # uniquely reconstructed from these samples.
    nyquist_hz = sample_hz / 2

    return {
        "sample_count": int(x.size),
        "finite_fraction": float(finite.mean()),
        "mean": float(np.mean(valid)),
        "rms": float(np.sqrt(np.mean(valid ** 2))),
        "peak_abs": float(np.max(np.abs(valid))),
        "sample_rate_hz": float(sample_hz),
        "nyquist_hz": float(nyquist_hz),
        "clipping_fraction": clipping_fraction,
        "warnings": warnings,
        "quality_ok": len(warnings) == 0,
    }


def check_timestamp_spacing(
    timestamps: np.ndarray | pd.Series,
    expected_sample_hz: float,
    *,
    tolerance_fraction: float = 0.05,
) -> dict:
    """Check timestamp ordering and approximate uniform sampling."""
    t = np.asarray(timestamps, dtype=float).reshape(-1)

    if expected_sample_hz <= 0:
        raise ValueError("expected_sample_hz must be positive")
    if not 0 <= tolerance_fraction < 1:
        raise ValueError("tolerance_fraction must be in [0, 1)")
    if t.size < 2 or not np.isfinite(t).all():
        raise ValueError("Need at least two finite timestamps")

    dt = np.diff(t)
    expected_dt = 1.0 / expected_sample_hz
    strictly_increasing = bool(np.all(dt > 0))
    uniform = bool(
        strictly_increasing
        and np.all(np.abs(dt - expected_dt) <= expected_dt * tolerance_fraction)
    )

    warnings = []
    if not strictly_increasing:
        warnings.append("timestamps_not_strictly_increasing")
    elif not uniform:
        warnings.append("irregular_timestamp_spacing")

    return {
        "strictly_increasing": strictly_increasing,
        "uniform_spacing": uniform,
        "median_interval_s": float(np.median(dt)),
        "expected_interval_s": expected_dt,
        "warnings": warnings,
        "quality_ok": len(warnings) == 0,
    }
