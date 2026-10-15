"""Explicit-metadata vibration signal I/O, preprocessing, and windowing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class VibrationSignal:
    samples: np.ndarray
    sample_rate_hz: float
    unit: str
    source: str


def load_signal(path: str | Path, sample_rate_hz: float, unit: str, column: str | None = None) -> VibrationSignal:
    """Load one signal; rate and physical unit must come from trusted metadata."""
    path = Path(path)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be a positive finite value")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("unit is required; do not infer or invent signal units")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = column or (archive.files[0] if len(archive.files) == 1 else None)
            if key is None or key not in archive.files:
                raise ValueError("NPZ with multiple arrays requires --column")
            values = archive[key]
    elif suffix in (".csv", ".txt"):
        frame = pd.read_csv(path)
        if column is None:
            numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric) != 1:
                raise ValueError("CSV must have one numeric column or specify --column")
            column = numeric[0]
        if column not in frame:
            raise ValueError(f"Signal column {column!r} not found in {path}")
        values = frame[column].to_numpy()
    else:
        raise ValueError(f"Unsupported signal format {suffix!r}; use CSV, TXT, NPY, or NPZ")
    values = np.asarray(values, dtype=np.float64).squeeze()
    if values.ndim != 1 or values.size < 16:
        raise ValueError(f"Expected a one-dimensional signal with >=16 samples: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Signal contains NaN or infinity: {path}")
    return VibrationSignal(values, float(sample_rate_hz), unit.strip(), str(path))


def preprocess(signal: VibrationSignal, low_hz: float | None = None, high_hz: float | None = None) -> VibrationSignal:
    """Optional zero-phase Butterworth filter with Nyquist validation."""
    nyquist = signal.sample_rate_hz / 2.0
    low = low_hz
    high = high_hz
    if low is None and high is None:
        return signal
    if low is not None and (not 0 < low < nyquist):
        raise ValueError("low_hz must be positive and below Nyquist")
    if high is not None and (not 0 < high < nyquist):
        raise ValueError("high_hz must be positive and below Nyquist")
    if low is not None and high is not None:
        if low >= high:
            raise ValueError("low_hz must be less than high_hz")
        wn, btype = [low / nyquist, high / nyquist], "bandpass"
    elif low is not None:
        wn, btype = low / nyquist, "highpass"
    else:
        wn, btype = high / nyquist, "lowpass"
    sos = butter(4, wn, btype=btype, output="sos")
    filtered = sosfiltfilt(sos, signal.samples)
    return VibrationSignal(filtered, signal.sample_rate_hz, signal.unit, signal.source)


def segment_signal(signal: VibrationSignal, window_seconds: float, overlap: float = 0.0) -> list[np.ndarray]:
    """Window a recording without crossing files; default windows do not overlap."""
    if window_seconds <= 0 or not 0 <= overlap < 1:
        raise ValueError("window_seconds must be positive and overlap in [0, 1)")
    window = int(round(window_seconds * signal.sample_rate_hz))
    step = int(round(window * (1 - overlap)))
    if window < 16 or signal.samples.size < window:
        raise ValueError("Recording is shorter than one full window")
    return [signal.samples[start : start + window] for start in range(0, signal.samples.size - window + 1, step)]
