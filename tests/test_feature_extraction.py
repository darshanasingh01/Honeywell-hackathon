import numpy as np
import pandas as pd
import pytest

from simulator.engine_simulator import simulate_engine
from feature_extraction.signal_features import (
    extract_time_features,
    extract_frequency_features,
    extract_band_features,
    extract_envelope_features,
    extract_fast_channel_features,
    extract_crank_features,
    extract_telemetry_features,
    crank_gated_knock,
    normalize_spectrogram_db,
    FAST_BANDS,
    TELEMETRY_SIGNALS,
)
from feature_extraction.pipeline import (
    FEATURE_COLUMNS,
    META_COLUMNS,
    iter_windows,
    compute_spectrogram,
    features_for_window,
    process_run,
    FS_EXPECTED,
    NPERSEG,
)

FS = 25600


def _tone(freq, n=12800, fs=FS, amp=1.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq * t)


# ---- pure feature functions ----
def test_time_features_keys_no_mean():
    out = extract_time_features(_tone(1000))
    assert set(out) == {"std", "rms", "peak", "crest_factor",
                        "kurtosis", "zero_crossing_rate"}
    assert "mean" not in out


def test_frequency_dominant():
    out = extract_frequency_features(_tone(3000), FS)
    assert abs(out["dominant_frequency_hz"] - 3000) < 300


def test_band_features_count_and_ratio_sum():
    out = extract_band_features(_tone(7000), FS)
    assert len(out) == 6  # 3 bands x (energy, ratio)
    ratios = sum(out[f"{b}_ratio"] for b in FAST_BANDS)
    assert ratios <= 1.0 + 1e-6
    # a 7 kHz tone lands in the knock band
    assert out["knock_5k_10k_ratio"] > out["low_0_1k_ratio"]


def test_fast_channel_feature_count():
    out = extract_fast_channel_features(_tone(2000), FS, "mic")
    assert len(out) == 19
    assert all(k.startswith("mic_") for k in out)


def test_envelope_keys():
    out = extract_envelope_features(_tone(7000), FS)
    assert set(out) == {"env_mean", "env_std", "env_peak",
                        "env_peak_to_mean", "env_repetition_hz"}


def test_telemetry_feature_count_and_slope():
    df = pd.DataFrame({
        "timestamp_s": np.linspace(0, 0.5, 5),
        "engine_rpm": [1500, 1510, 1520, 1530, 1540],
        "coolant_temp_c": [90, 91, 92, 93, 94],
        "throttle_pct": [30, 30, 30, 30, 30],
        "engine_load": [0.3, 0.3, 0.3, 0.3, 0.3],
        "stft_pct": [0, 0, 0, 0, 0],
        "ltft_pct": [2, 2, 2, 2, 2],
        "ignition_timing": [15, 15, 15, 15, 15],
    })
    out = extract_telemetry_features(df)
    assert len(out) == 35  # 7 signals x 5 stats
    assert out["tel_coolant_temp_c_slope"] > 0


def test_crank_features():
    crank = np.zeros(FS)
    crank[::512] = 1  # rising edges every 512 samples -> 50 rev/s -> 3000 rpm
    out = extract_crank_features(crank, FS)
    assert abs(out["crank_rpm"] - 3000) < 50


def test_normalize_spectrogram_db_zero_median():
    spec = np.random.RandomState(0).randn(65, 40).astype(np.float32) * 5 + 20
    norm = normalize_spectrogram_db(spec)
    assert np.allclose(np.median(norm, axis=0), 0.0, atol=1e-4)


# ---- pipeline window machinery ----
def test_iter_windows_overlap():
    wins = list(iter_windows(256000, FS))
    win = int(0.5 * FS)
    hop = int(0.25 * FS)
    assert wins[0] == (0, win)
    assert wins[1][0] - wins[0][0] == hop  # 50% overlap
    # last window fully inside the signal
    assert wins[-1][1] <= 256000


def test_compute_spectrogram_shape():
    mic = _tone(2000)
    knock = _tone(7000)
    spec = compute_spectrogram(mic, knock, FS, nperseg=NPERSEG)
    assert spec.ndim == 3
    assert spec.shape[0] == 2       # channels
    assert spec.shape[1] == NPERSEG // 2 + 1
    assert spec.dtype == np.float32


def test_features_for_window_schema():
    telemetry, waveform = simulate_engine("knock", duration_s=0.6, seed=0)
    win = int(0.5 * FS)
    mic = waveform["microphone_audio"].to_numpy()[:win]
    knock = waveform["knock_sensor"].to_numpy()[:win]
    crank = waveform["crank_pulse"].to_numpy()[:win]
    ax = waveform["accel_x"].to_numpy()[:win]
    ay = waveform["accel_y"].to_numpy()[:win]
    az = waveform["accel_z"].to_numpy()[:win]
    feat, spec = features_for_window(mic, knock, crank, telemetry, ax, ay, az, FS)
    assert set(feat.keys()) == set(FEATURE_COLUMNS)
    assert len(feat) == 136
    # the three accelerometer axes each contribute 19 prefixed fast-channel keys
    for prefix in ("accel_x", "accel_y", "accel_z"):
        assert sum(k.startswith(prefix + "_") for k in feat) == 19
    assert spec.shape[0] == 2  # spectrogram stays 2-channel (mic, knock)


def test_fs_gate_rejects_wrong_rate():
    # pipeline main derives fs from timestamps and must reject != 25600
    from feature_extraction.pipeline import _derive_fs
    ts = np.arange(0, 1, 1 / 4000)
    assert not np.isclose(_derive_fs(ts), FS_EXPECTED, rtol=0.01)


def test_process_run_alignment_and_straddle_skip():
    telemetry, waveform = simulate_engine("normal", duration_s=2.0, seed=0)
    rows, specs = process_run(waveform, telemetry, "normal_s0", "normal", FS)
    assert len(rows) == len(specs)
    # window_ids unique and labels consistent
    wids = [r["window_id"] for r in rows]
    assert len(wids) == len(set(wids))
    assert all(r["label"] == "normal" for r in rows)

    # straddle: a window crossing two labels is skipped
    mixed = waveform.copy()
    half = len(mixed) // 2
    cond = np.array(["normal"] * len(mixed), dtype=object)
    cond[half:] = "knock"
    mixed["condition"] = cond
    rows2, specs2 = process_run(mixed, telemetry, "mixed_s0", "mixed", FS)
    # every retained window has a single pure label
    assert all(r["label"] in ("normal", "knock") for r in rows2)
    assert len(rows2) == len(specs2)
    # fewer windows than the single-label run (straddling ones dropped)
    assert len(rows2) < len(rows)
