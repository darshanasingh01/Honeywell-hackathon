"""Pure DSP feature-extraction functions for engine diagnostic waveforms.

No file I/O. Every function takes raw arrays (plus the sample rate where needed)
and returns an ordered dict of named features. These compose into the 136-feature
contract consumed by the pipeline.

Scipy is used for the Butterworth band-pass (``butter(output='sos')`` +
``sosfiltfilt``), the Hilbert envelope and Fisher kurtosis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis as scipy_kurtosis

# --- Contract constants ---
FAST_BANDS = {
    "low_0_1k": (0.0, 1000.0),
    "mid_1k_5k": (1000.0, 5000.0),
    "knock_5k_10k": (5000.0, 10000.0),
}
TELEMETRY_SIGNALS = [
    "engine_rpm",
    "coolant_temp_c",
    "throttle_pct",
    "engine_load",
    "stft_pct",
    "ltft_pct",
    "ignition_timing",
]
ENVELOPE_BAND = (5000.0, 10000.0)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _bandpass(signal, fs, lo, hi, order=4):
    """Zero-phase Butterworth band-pass via second-order sections."""
    signal = np.asarray(signal, dtype=float)
    nyq = fs / 2.0
    lo = max(lo, 1.0)
    hi = min(hi, nyq * 0.999)
    if hi <= lo:
        return np.zeros_like(signal)
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    # sosfiltfilt needs length > 3 * (order of filter padding)
    padlen = 3 * (sos.shape[0] * 2)
    if signal.size <= padlen:
        return np.zeros_like(signal)
    return sosfiltfilt(sos, signal)


def _rising_edges(crank):
    """Sample indices where crank goes 0 -> (>0)."""
    crank = np.asarray(crank, dtype=float)
    hi = crank > 0.5
    if hi.size < 2:
        return np.array([], dtype=int)
    return np.nonzero((~hi[:-1]) & (hi[1:]))[0] + 1


# ----------------------------------------------------------------------------
# Time-domain
# ----------------------------------------------------------------------------
def extract_time_features(signal):
    """Volume-robust time-domain stats (no mean: AC signal)."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        raise ValueError("Signal window cannot be empty.")

    std = float(np.std(signal))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    crest = float(peak / rms) if rms > 0 else 0.0
    kurt = float(scipy_kurtosis(signal, fisher=True, bias=False)) if signal.size > 3 else 0.0
    if not np.isfinite(kurt):
        kurt = 0.0
    zcr = float(np.mean(signal[:-1] * signal[1:] < 0)) if signal.size > 1 else 0.0

    return {
        "std": std,
        "rms": rms,
        "peak": peak,
        "crest_factor": crest,
        "kurtosis": kurt,
        "zero_crossing_rate": zcr,
    }


# ----------------------------------------------------------------------------
# Frequency-domain
# ----------------------------------------------------------------------------
def extract_frequency_features(signal, fs):
    """Dominant frequency (DC excluded) and spectral centroid."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        raise ValueError("Signal window cannot be empty.")
    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")

    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / fs)

    if spectrum.size > 1 and np.any(spectrum[1:] > 0):
        dominant = float(freqs[1 + int(np.argmax(spectrum[1:]))])
    else:
        dominant = 0.0

    total = float(np.sum(spectrum))
    centroid = float(np.sum(freqs * spectrum) / total) if total > 0 else 0.0

    return {
        "dominant_frequency_hz": dominant,
        "spectral_centroid_hz": centroid,
    }


# ----------------------------------------------------------------------------
# Band energies
# ----------------------------------------------------------------------------
def extract_band_features(signal, fs, bands=FAST_BANDS):
    """Per-band energy (sum of squared magnitude, DC excluded) and ratio."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        raise ValueError("Signal window cannot be empty.")

    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / fs)
    power = spectrum ** 2
    power[0] = 0.0  # exclude DC
    total = float(np.sum(power))

    out = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        energy = float(np.sum(power[mask]))
        out[f"{name}_energy"] = energy
        out[f"{name}_ratio"] = float(energy / total) if total > 0 else 0.0
    return out


# ----------------------------------------------------------------------------
# Envelope (demodulated high-frequency band)
# ----------------------------------------------------------------------------
def extract_envelope_features(signal, fs, band=ENVELOPE_BAND):
    """Hilbert envelope of the band-passed signal + its repetition rate."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        raise ValueError("Signal window cannot be empty.")

    filtered = _bandpass(signal, fs, band[0], band[1])
    env = np.abs(hilbert(filtered))

    env_mean = float(np.mean(env))
    env_std = float(np.std(env))
    env_peak = float(np.max(env)) if env.size else 0.0
    env_p2m = float(env_peak / env_mean) if env_mean > 0 else 0.0

    # Repetition rate: dominant frequency of the mean-removed envelope.
    env_c = env - env_mean
    if env_c.size > 1 and np.any(np.abs(env_c) > 0):
        espec = np.abs(np.fft.rfft(env_c))
        efreq = np.fft.rfftfreq(len(env_c), d=1.0 / fs)
        rep = float(efreq[1 + int(np.argmax(espec[1:]))]) if espec.size > 1 else 0.0
    else:
        rep = 0.0

    return {
        "env_mean": env_mean,
        "env_std": env_std,
        "env_peak": env_peak,
        "env_peak_to_mean": env_p2m,
        "env_repetition_hz": rep,
    }


# ----------------------------------------------------------------------------
# Composite fast-channel features (19 keys, prefixed)
# ----------------------------------------------------------------------------
def extract_fast_channel_features(signal, fs, prefix):
    """time (6) + freq (2) + band (6) + envelope (5) = 19 prefixed features."""
    out = {}
    for name, value in extract_time_features(signal).items():
        out[f"{prefix}_{name}"] = value
    for name, value in extract_frequency_features(signal, fs).items():
        out[f"{prefix}_{name}"] = value
    for name, value in extract_band_features(signal, fs).items():
        out[f"{prefix}_{name}"] = value
    for name, value in extract_envelope_features(signal, fs).items():
        out[f"{prefix}_{name}"] = value
    return out


# ----------------------------------------------------------------------------
# Crank features
# ----------------------------------------------------------------------------
def extract_crank_features(crank, fs):
    """Rotational-speed stats from crank rising edges."""
    edges = _rising_edges(crank)
    if edges.size < 2:
        return {
            "crank_rpm": 0.0,
            "crank_rpm_std": 0.0,
            "crank_interval_cv": 0.0,
        }
    intervals = np.diff(edges) / fs  # seconds per revolution
    rpms = 60.0 / intervals
    mean_int = float(np.mean(intervals))
    cv = float(np.std(intervals) / mean_int) if mean_int > 0 else 0.0
    return {
        "crank_rpm": float(np.mean(rpms)),
        "crank_rpm_std": float(np.std(rpms)),
        "crank_interval_cv": cv,
    }


# ----------------------------------------------------------------------------
# Telemetry features (35 keys)
# ----------------------------------------------------------------------------
def extract_telemetry_features(telemetry_window: pd.DataFrame, signals=TELEMETRY_SIGNALS):
    """Per-signal mean/std/min/max/slope over the telemetry window."""
    out = {}
    if telemetry_window is not None and len(telemetry_window) > 0 \
            and "timestamp_s" in telemetry_window.columns:
        tvals = telemetry_window["timestamp_s"].to_numpy(dtype=float)
        t0 = tvals - tvals[0]
    else:
        tvals = None
        t0 = None

    for sig in signals:
        if telemetry_window is None or len(telemetry_window) == 0 \
                or sig not in telemetry_window.columns:
            out[f"tel_{sig}_mean"] = 0.0
            out[f"tel_{sig}_std"] = 0.0
            out[f"tel_{sig}_min"] = 0.0
            out[f"tel_{sig}_max"] = 0.0
            out[f"tel_{sig}_slope"] = 0.0
            continue
        v = telemetry_window[sig].to_numpy(dtype=float)
        out[f"tel_{sig}_mean"] = float(np.mean(v))
        out[f"tel_{sig}_std"] = float(np.std(v))
        out[f"tel_{sig}_min"] = float(np.min(v))
        out[f"tel_{sig}_max"] = float(np.max(v))
        if v.size >= 2 and np.ptp(t0) > 0:
            A = np.vstack([t0, np.ones_like(t0)]).T
            slope = float(np.linalg.lstsq(A, v, rcond=None)[0][0])
        else:
            slope = 0.0
        out[f"tel_{sig}_slope"] = slope
    return out


# ----------------------------------------------------------------------------
# Crank-gated knock detection
# ----------------------------------------------------------------------------
def crank_gated_knock(knock, crank, fs, window_ms=(0.5, 3.0),
                      knock_band=ENVELOPE_BAND, intensity_thresh=None):
    """5-10 kHz energy in the post-firing window after each crank edge.

    Returns mean/max per-cycle intensity and the fraction of cycles whose
    intensity exceeds an adaptive threshold (median + k*MAD).
    """
    knock = np.asarray(knock, dtype=float)
    edges = _rising_edges(crank)
    filtered = _bandpass(knock, fs, knock_band[0], knock_band[1])

    lo = int(round(window_ms[0] / 1000.0 * fs))
    hi = int(round(window_ms[1] / 1000.0 * fs))

    intensities = []
    for e in edges:
        a = e + lo
        b = e + hi
        if b > filtered.size:
            continue
        seg = filtered[a:b]
        if seg.size == 0:
            continue
        intensities.append(float(np.sqrt(np.mean(seg ** 2))))

    if not intensities:
        return {
            "knock_cycle_intensity_mean": 0.0,
            "knock_cycle_intensity_max": 0.0,
            "knock_cycle_pct_knocking": 0.0,
        }

    intensities = np.asarray(intensities)
    mean_i = float(np.mean(intensities))
    max_i = float(np.max(intensities))

    if intensity_thresh is None:
        med = np.median(intensities)
        mad = np.median(np.abs(intensities - med))
        thresh = med + 3.0 * 1.4826 * mad
        # Guard against degenerate (near-zero MAD) windows.
        if mad <= 1e-12:
            thresh = med * 1.5 + 1e-9
    else:
        thresh = intensity_thresh

    pct = float(np.mean(intensities > thresh))

    return {
        "knock_cycle_intensity_mean": mean_i,
        "knock_cycle_intensity_max": max_i,
        "knock_cycle_pct_knocking": pct,
    }


# ----------------------------------------------------------------------------
# Spectrogram dB normalization
# ----------------------------------------------------------------------------
def normalize_spectrogram_db(spec_db):
    """Per-window (per-time-column) dB normalization: subtract column median."""
    spec_db = np.asarray(spec_db, dtype=np.float32)
    if spec_db.size == 0:
        return spec_db
    med = np.median(spec_db, axis=0, keepdims=True)
    return (spec_db - med).astype(np.float32)
