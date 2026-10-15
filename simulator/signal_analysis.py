"""Frequency-domain tools for synthetic engine signals.

These functions analyse sampled signals; they do not independently diagnose faults.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram


def frequency_spectrum(
    signal: np.ndarray,
    sample_rate_hz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided Hann-window amplitude spectrum."""
    x = np.asarray(signal, dtype=float)

    if x.ndim != 1 or x.size < 16:
        raise ValueError("Signal must be 1-D and contain at least 16 samples")
    if not np.isfinite(x).all():
        raise ValueError("Signal contains non-finite values")
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive")

    x = x - np.mean(x)
    window = np.hanning(len(x))
    scale = np.sum(window)

    if scale <= 0:
        raise ValueError("Unable to construct analysis window")

    spectrum = np.abs(np.fft.rfft(x * window)) / scale
    if len(x) > 1:
        spectrum[1:-1 if len(x) % 2 == 0 else None] *= 2

    frequencies = np.fft.rfftfreq(len(x), d=1.0 / sample_rate_hz)
    return frequencies, spectrum


def time_frequency_spectrogram(
    signal: np.ndarray,
    sample_rate_hz: int,
    *,
    nperseg: int = 512,
    overlap_fraction: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, frequency and power-density arrays for a spectrogram."""
    x = np.asarray(signal, dtype=float)

    if x.ndim != 1 or x.size < 16:
        raise ValueError("Signal must be 1-D and contain at least 16 samples")
    if not np.isfinite(x).all():
        raise ValueError("Signal contains non-finite values")
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")

    segment_length = min(int(nperseg), len(x))
    if segment_length < 16:
        raise ValueError("Analysis segment must contain at least 16 samples")

    overlap = min(
        int(segment_length * overlap_fraction),
        segment_length - 1,
    )

    frequencies, times, power = spectrogram(
        x,
        fs=sample_rate_hz,
        window="hann",
        nperseg=segment_length,
        noverlap=overlap,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    return times, frequencies, power
