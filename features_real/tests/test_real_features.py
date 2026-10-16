import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_features import audio_features as af            # noqa: E402
from real_features import csv_features as cf              # noqa: E402
from real_features.config import AudioConfig, CsvConfig   # noqa: E402
from real_features.manifest import assign_recording_splits, build_manifest  # noqa: E402
from real_features.pipeline import run_audio, run_csv     # noqa: E402

SR = 44100


def tone(freq, dur=1.0, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def write_wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


# ------------------------------------------------------------------ windowing
def test_window_starts_full_and_short():
    starts, padded = af.window_starts(SR * 3, SR, SR // 2, SR // 2)
    assert starts == [0, 22050, 44100, 66150, 88200] and not padded
    starts, padded = af.window_starts(int(0.6 * SR), SR, SR // 2, SR // 2)
    assert starts == [0] and padded
    assert af.window_starts(int(0.3 * SR), SR, SR // 2, SR // 2) == ([], False)
    assert len(af.cut(np.ones(1000), 0, 2000)) == 2000


# ------------------------------------------------------------------ spectral features
def test_mel_filterbank_has_no_empty_bands():
    cfg = AudioConfig()
    fb = af.mel_filterbank(cfg.sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    assert fb.shape == (cfg.n_mels, cfg.n_fft // 2 + 1)
    assert (fb.sum(axis=1) > 0).all()


def feats_of(x):
    cfg = AudioConfig()
    fb = af.mel_filterbank(cfg.sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    hp = af.highpass(x, SR, cfg.highpass_hz)
    bp = af.bandpass(hp, SR, *cfg.env_band)
    lm = af.log_mel(hp, SR, fb, cfg.n_fft, cfg.spec_hop)
    return af.window_features(hp, bp, SR, lm, cfg), lm, cfg


def test_pure_tone_features():
    f, lm, cfg = feats_of(tone(440.0))
    assert abs(f["dominant_hz"] - 440.0) <= 1.0
    assert abs(f["spec_centroid"] - 440.0) < 20.0
    assert abs(f["rms"] - 0.3 / np.sqrt(2)) < 0.01
    assert lm.shape == (cfg.n_mels, 1 + (SR - cfg.n_fft) // cfg.spec_hop)
    assert set(af.feature_names(cfg)) <= set(f)      # every advertised column exists
    assert all(np.isfinite(v) for v in f.values())


def test_envelope_detects_modulation_frequency():
    t = np.arange(SR) / SR
    x = 0.3 * (1 + 0.8 * np.sin(2 * np.pi * 30 * t)) * np.sin(2 * np.pi * 3000 * t)
    f, _, _ = feats_of(x)
    assert abs(f["env_peak_hz"] - 30.0) <= 1.0
    assert f["env_peak_ratio"] > 10


def test_order_spectrum_peaks_at_orders():
    rpm = 1500.0                              # 25 Hz rotation
    x = tone(25.0) + 0.5 * tone(50.0)         # 1st and 2nd order
    orders, mag = af.order_spectrum(x, SR, rpm, max_order=6, n_bins=601)
    assert abs(orders[np.argmax(mag)] - 1.0) < 0.02
    m2 = (orders > 1.5) & (orders < 2.5)
    assert abs(orders[m2][np.argmax(mag[m2])] - 2.0) < 0.02


# ------------------------------------------------------------------ manifest + splits
def make_tree(tmp_path):
    root = tmp_path / "audio"
    for i in range(10):
        write_wav(root / "p1_diesel_part1" / f"REC{i + 1}.wav", tone(100 + i))
    for i in range(10, 16):
        write_wav(root / "PETROL" / f"REC{i + 1}.wav", tone(300 + i))
    # copies of a petrol file in an unrelated folder (must be ignored) and an identical diesel copy (dropped)
    write_wav(root / "ilove" / "REC11_1.wav", tone(310))
    write_wav(root / "diesel_extra" / "REC1.wav", tone(100))
    return root


def test_manifest_labels_skips_and_dedupes(tmp_path):
    man, rep = build_manifest(make_tree(tmp_path))
    assert len(man) == 16
    assert rep["skipped_unlabelled_folder"] == 1 and rep["duplicate_files_dropped"] == 1
    assert (man.fuel.value_counts().to_dict()) == {"diesel": 10, "petrol": 6}
    assert set(man.label) == {0, 1}


def test_manifest_conflicting_fuel_raises(tmp_path):
    root = tmp_path / "a"
    write_wav(root / "diesel" / "REC1.wav", tone(100))
    write_wav(root / "petrol" / "REC2.wav", tone(100))     # same audio, other fuel
    with pytest.raises(ValueError):
        build_manifest(root)


def test_recording_splits_are_disjoint_and_stratified(tmp_path):
    rows = [{"recording_key": f"r{i}", "fuel": "diesel" if i < 40 else "petrol"} for i in range(60)]
    s = assign_recording_splits(pd.DataFrame(rows), 0.15, 0.15, seed=1)
    assert s.recording_key.is_unique
    for fuel in ("diesel", "petrol"):
        assert set(s[s.fuel == fuel].split) == {"train", "val", "test"}
    again = assign_recording_splits(pd.DataFrame(rows), 0.15, 0.15, seed=1)
    assert (s.split == again.split).all()           # deterministic


# ------------------------------------------------------------------ CSV
def make_csv_df(per_class=6000, classes=(0, 1, 2, 3)):
    rng = np.random.default_rng(0)
    parts = []
    for c in classes:
        base = np.cumsum(rng.normal(size=(per_class, 3)), axis=0) * 0.01 + c
        d = pd.DataFrame(base, columns=["MAP", "Consumption L/H", "RPM"])
        d.insert(0, "Fault", c)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_csv_windows_stay_inside_one_class_and_split():
    cfg = CsvConfig()
    df = make_csv_df()
    win, rep = cf.build_csv_windows(df, cfg)
    assert rep["rows_in_contiguous_runs_note"] == "contiguous"
    labels = df["Fault"].to_numpy()
    for w in win.itertuples():
        assert len(set(labels[w.row_start:w.row_end])) == 1
        assert labels[w.row_start] == w.fault
    ws = win.sort_values("row_start")
    sp, rs, re_ = ws.split.to_numpy(), ws.row_start.to_numpy(), ws.row_end.to_numpy()
    gaps = [rs[i + 1] - re_[i] for i in range(len(ws) - 1) if sp[i] != sp[i + 1]]
    assert min(gaps) >= 2 * cfg.purge_rows - cfg.win_rows or min(gaps) >= cfg.purge_rows
    assert set(win.split) == {"train", "val", "test"}
    assert np.isfinite(win.select_dtypes("number").to_numpy()).all()
    assert "consumption_l_h_mean" in win.columns


def test_csv_slope_is_exact_for_a_ramp():
    n = 6000
    df = pd.DataFrame({"Fault": 0, "X": 0.5 * np.arange(n, dtype=float)})
    win, _ = cf.build_csv_windows(df, CsvConfig())
    assert np.allclose(win["x_slope"], 0.5)
    assert np.allclose(win["x_std"], win["x_std"].iloc[0])


def test_csv_shuffled_rows_are_rejected_with_clear_message():
    df = make_csv_df(per_class=3000)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="interleaved"):
        cf.build_csv_windows(df, CsvConfig())


def test_csv_partly_interleaved_rows_get_a_warning():
    df = make_csv_df(per_class=6000)
    df.loc[100:110, "Fault"] = 3          # a few stray rows inside class 0
    _, rep = cf.build_csv_windows(df, CsvConfig())
    assert rep["rows_in_contiguous_runs_note"].startswith("WARNING")


def test_csv_duplicate_row_dropped(tmp_path):
    df = make_csv_df(per_class=50, classes=(0, 1))
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    out, rep = cf.load_csv(p)
    assert rep["exact_duplicate_rows_dropped"] == 1 and len(out) == len(df) - 1


# ------------------------------------------------------------------ end to end
def test_end_to_end_outputs_align(tmp_path):
    root = make_tree(tmp_path)
    out = tmp_path / "out"
    cfg = AudioConfig()
    win, man, summ = run_audio(root, out, cfg, log=lambda *_: None)
    assert sorted(p.name for p in out.iterdir()) == [
        "real_audio_manifest.csv", "real_audio_spectrograms.npz", "real_audio_summary.json", "real_audio_windows.csv"]
    z = np.load(out / "real_audio_spectrograms.npz", allow_pickle=True)
    assert z["spectrograms"].shape[0] == len(win) == len(pd.read_csv(out / "real_audio_windows.csv"))
    assert (z["window_id"] == win.window_id.to_numpy()).all()
    assert win.groupby("recording_key").split.nunique().max() == 1
    assert json.loads((out / "real_audio_summary.json").read_text())["n_recordings"] == 16
    # CSV run writes only real_csv_* files
    p = tmp_path / "e.csv"
    make_csv_df().to_csv(p, index=False)
    run_csv(p, out, CsvConfig(), log=lambda *_: None)
    assert all(f.startswith("real_") for f in os.listdir(out))
