
"""Signal feature extraction for synthetic engine diagnostic waveforms."""

import numpy as np


def extract_time_features(signal):
    """Calculate basic statistical features from a signal window."""
    signal = np.asarray(signal, dtype=float)

    if signal.size == 0:
        raise ValueError("Signal window cannot be empty.")

    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    mean = np.mean(signal)
    std = np.std(signal)
    zero_crossings = np.mean(signal[:-1] * signal[1:] < 0) if signal.size > 1 else 0.0

    return {
        "mean": float(mean),
        "std": float(std),
        "rms": float(rms),
        "peak": float(peak),
        "crest_factor": float(peak / rms) if rms > 0 else 0.0,
        "zero_crossing_rate": float(zero_crossings),
    }


def extract_frequency_features(signal, sampling_rate):
    """Calculate frequency-domain features using the real FFT."""
    signal = np.asarray(signal, dtype=float)

    if signal.size == 0:
        raise ValueError("Signal window cannot be empty.")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    # Remove DC offset before calculating the spectrum.
    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sampling_rate)

    # Ignore the zero-frequency component when finding the dominant frequency.
    if spectrum.size > 1 and np.any(spectrum[1:] > 0):
        dominant_index = 1 + np.argmax(spectrum[1:])
        dominant_frequency = frequencies[dominant_index]
    else:
        dominant_frequency = 0.0

    total_magnitude = np.sum(spectrum)
    spectral_centroid = (
        np.sum(frequencies * spectrum) / total_magnitude
        if total_magnitude > 0 else 0.0
    )

    # The Nyquist limit at 4 kHz sampling is 2 kHz.
    # This band is suitable for the simulator's 1.1 kHz knock component.
    knock_band = (frequencies >= 700) & (frequencies <= 1800)
    knock_band_energy = float(np.sum(spectrum[knock_band] ** 2))

    return {
        "dominant_frequency_hz": float(dominant_frequency),
        "spectral_centroid_hz": float(spectral_centroid),
        "knock_band_energy_700_1800_hz": knock_band_energy,
    }


def extract_signal_features(signal, sampling_rate, prefix):
    """Combine time-domain and frequency-domain features."""
    features = {}

    for name, value in extract_time_features(signal).items():
        features[f"{prefix}_{name}"] = value

    for name, value in extract_frequency_features(signal, sampling_rate).items():
        features[f"{prefix}_{name}"] = value

    return features
