"""Simplified, angle-based crankshaft position sensor model.

This is a configurable synthetic reference model, not a calibrated
representation of a particular production engine or sensor.
"""
from __future__ import annotations

import numpy as np


def generate_ckp(
    crank_angle_deg: np.ndarray,
    signal_hz: int,
    *,
    nominal_teeth: int = 60,
    missing_teeth: int = 2,
) -> np.ndarray:
    """Return a sampled binary CKP waveform from unwrapped crank angle."""
    angle = np.asarray(crank_angle_deg, dtype=float)

    if angle.ndim != 1 or len(angle) < 2:
        raise ValueError(
            "crank_angle_deg must be a 1-D array with at least 2 samples"
        )
    if not np.isfinite(angle).all():
        raise ValueError("crank angles must be finite")
    if np.any(np.diff(angle) <= 0):
        raise ValueError("crank angles must be strictly increasing")
    if (
        isinstance(signal_hz, (bool, np.bool_))
        or not isinstance(signal_hz, (int, np.integer))
        or signal_hz <= 0
    ):
        raise ValueError("signal_hz must be a positive integer")
    if (
        isinstance(nominal_teeth, (bool, np.bool_))
        or not isinstance(nominal_teeth, (int, np.integer))
        or isinstance(missing_teeth, (bool, np.bool_))
        or not isinstance(missing_teeth, (int, np.integer))
        or nominal_teeth < 4
        or not 1 <= missing_teeth < nominal_teeth - 1
    ):
        raise ValueError("invalid tooth-wheel configuration")

    tooth_pitch = 360.0 / nominal_teeth
    tooth_index = np.floor(angle / tooth_pitch).astype(np.int64)
    tooth_slot = tooth_index % nominal_teeth
    valid_tooth = tooth_slot < nominal_teeth - missing_teeth

    # A pulse is emitted when a sampled angle first enters a new tooth slot.
    # Finite sampling can miss tooth boundaries at insufficient sample rates.
    crossed_boundary = np.r_[True, np.diff(tooth_index) > 0]
    return (valid_tooth & crossed_boundary).astype(np.uint8)
