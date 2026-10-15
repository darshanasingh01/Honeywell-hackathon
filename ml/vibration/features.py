"""Time, spectrum, and transient features for one vibration window."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew


def extract_features(samples: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or x.size < 16 or not np.isfinite(x).all():
        raise ValueError("Expected a finite one-dimensional signal window")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    centered = x - np.mean(x)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    peak = float(np.max(np.abs(centered)))
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(x.size)))
    power = spectrum ** 2
    frequencies = np.fft.rfftfreq(x.size, 1.0 / sample_rate_hz)
    total_power = float(np.sum(power)) + 1e-15
    normalized = power / total_power
    centroid = float(np.sum(frequencies * normalized))
    positive = normalized[normalized > 0]
    entropy = float(-np.sum(positive * np.log2(positive)) / np.log2(max(2, len(normalized))))
    peak_indices, _ = find_peaks(spectrum)
    if peak_indices.size:
        strongest = peak_indices[np.argsort(spectrum[peak_indices])[-3:][::-1]]
        strongest = strongest[np.argsort(spectrum[strongest])[::-1]]
    else:
        strongest = np.array([], dtype=int)
    peak_freqs = [float(frequencies[i]) for i in strongest]
    peak_amps = [float(spectrum[i]) for i in strongest]
    peak_freqs += [0.0] * (3 - len(peak_freqs))
    peak_amps += [0.0] * (3 - len(peak_amps))
    nyquist = sample_rate_hz / 2
    # Fixed fractional Nyquist bands are valid without assuming any dataset's sensor bandwidth.
    edges = [0.0, nyquist * .05, nyquist * .20, nyquist * .50, nyquist]
    band_energy = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high if high < nyquist else frequencies <= high)
        band_energy.append(float(np.sum(power[mask]) / total_power))
    # Impulse-sensitive: crest factor, excess kurtosis, robust peak ratio, and event density.
    abs_median = float(np.median(np.abs(centered))) + 1e-15
    mad = float(np.median(np.abs(centered - np.median(centered)))) + 1e-15
    absolute_centered = np.abs(centered)
    event_threshold = float(np.median(absolute_centered) + 8.0 * mad)
    event_count = int(find_peaks(absolute_centered, height=event_threshold)[0].size)
    result = {
        "rms": rms,
        "std": float(np.std(x)),
        "peak_to_peak": float(np.ptp(x)),
        "crest_factor": peak / (rms + 1e-15),
        "kurtosis_excess": float(kurtosis(x, fisher=True, bias=False)) if x.size > 4 and np.std(x) > 0 else 0.0,
        "skewness": float(skew(x, bias=False)) if x.size > 2 and np.std(x) > 0 else 0.0,
        "spectral_centroid_hz": centroid,
        "spectral_entropy": entropy,
        "impulse_peak_median_ratio": peak / abs_median,
        "impulse_event_rate_hz": event_count / (x.size / sample_rate_hz),
        "impulse_peak_count": float(event_count),
    }
    result.update({f"fft_peak_{i + 1}_hz": peak_freqs[i] for i in range(3)})
    result.update({f"fft_peak_{i + 1}_amplitude": peak_amps[i] for i in range(3)})
    result.update({f"band_energy_fraction_{i + 1}": band_energy[i] for i in range(4)})
    return result


def feature_names() -> list[str]:
    return [
        "rms", "std", "peak_to_peak", "crest_factor", "kurtosis_excess", "skewness",
        "spectral_centroid_hz", "spectral_entropy", "impulse_peak_median_ratio",
        "impulse_event_rate_hz", "impulse_peak_count",
        "fft_peak_1_hz", "fft_peak_2_hz", "fft_peak_3_hz",
        "fft_peak_1_amplitude", "fft_peak_2_amplitude", "fft_peak_3_amplitude",
        "band_energy_fraction_1", "band_energy_fraction_2", "band_energy_fraction_3", "band_energy_fraction_4",
    ]
