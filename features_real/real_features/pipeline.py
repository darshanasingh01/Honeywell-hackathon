"""End-to-end runs. Every output file starts with 'real_' so nothing can overwrite the synthetic set."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import audio_features as af
from . import csv_features as cf
from .config import AudioConfig, CsvConfig, as_dict
from .manifest import assign_recording_splits, build_manifest


def run_audio(audio_root, out_dir, cfg=None, save_spectrograms=True, log=print):
    cfg = cfg or AudioConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest, mrep = build_manifest(audio_root)
    manifest = assign_recording_splits(manifest, cfg.val_frac, cfg.test_frac, cfg.seed)
    log(f"[audio] {len(manifest)} unique recordings "
        f"({manifest.fuel.value_counts().to_dict()}); skipped unlabelled={mrep['skipped_unlabelled_folder']}, "
        f"duplicates dropped={mrep['duplicate_files_dropped']}")

    win = int(round(cfg.win_s * cfg.sr))
    hop = int(round(cfg.hop_s * cfg.sr))
    min_len = int(round(cfg.min_len_s * cfg.sr))
    fb = af.mel_filterbank(cfg.sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)

    feat_rows, specs, too_short, durations = [], [], [], []
    t0 = time.time()
    for i, rec in enumerate(manifest.itertuples(index=False)):
        x = af.read_wav(rec.path, cfg.sr)
        durations.append(len(x) / cfg.sr)
        x_hp = af.highpass(x, cfg.sr, cfg.highpass_hz)
        x_bp = af.bandpass(x_hp, cfg.sr, *cfg.env_band)
        starts, padded = af.window_starts(len(x), win, hop, min_len)
        if not starts:
            too_short.append(rec.recording_key)
            continue
        for k, s in enumerate(starts):
            w_hp, w_bp = af.cut(x_hp, s, win), af.cut(x_bp, s, win)
            lm = af.log_mel(w_hp, cfg.sr, fb, cfg.n_fft, cfg.spec_hop)
            f = af.window_features(w_hp, w_bp, cfg.sr, lm, cfg)
            row = {
                "window_id": f"{rec.recording_key}_w{k:03d}",
                "recording_key": rec.recording_key, "rec_num": rec.rec_num,
                "fuel": rec.fuel, "label": rec.label, "split": rec.split,
                "start_s": s / cfg.sr, "end_s": (s + win) / cfg.sr, "padded": int(padded),
                "clipped": int(np.abs(x).max() >= 0.999),
            }
            row.update(f)
            feat_rows.append(row)
            if save_spectrograms:
                specs.append(lm.astype(np.float16))
        if (i + 1) % 50 == 0:
            log(f"[audio] {i + 1}/{len(manifest)} recordings ({time.time() - t0:.0f}s)")

    win_df = pd.DataFrame(feat_rows)
    manifest.drop(columns=["sha1"]).to_csv(out / "real_audio_manifest.csv", index=False)
    win_df.to_csv(out / "real_audio_windows.csv", index=False)
    if save_spectrograms:
        np.savez_compressed(out / "real_audio_spectrograms.npz",
                            spectrograms=np.stack(specs), window_id=win_df["window_id"].to_numpy(),
                            note="log-mel dB, shape (n_windows, n_mels, n_frames); rows align with real_audio_windows.csv")
    summary = {
        "config": as_dict(cfg), "manifest_report": mrep,
        "n_recordings": int(len(manifest)), "n_windows": int(len(win_df)),
        "too_short_dropped": too_short,
        "windows_per_fuel": win_df["fuel"].value_counts().to_dict(),
        "windows_per_split": win_df["split"].value_counts().to_dict(),
        "recordings_per_split": manifest["split"].value_counts().to_dict(),
        "duration_s": {"min": float(np.min(durations)), "mean": float(np.mean(durations)),
                       "max": float(np.max(durations))},
        "spectrogram_shape": list(specs[0].shape) if specs else None,
        "n_feature_columns": len(af.feature_names(cfg)),
    }
    (out / "real_audio_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"[audio] {len(win_df)} windows -> {out}")
    return win_df, manifest, summary


def run_csv(csv_path, out_dir, cfg=None, log=print):
    cfg = cfg or CsvConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, rep = cf.load_csv(csv_path, cfg.label_col)
    win_df, wrep = cf.build_csv_windows(df, cfg)
    win_df.to_csv(out / "real_csv_windows.csv", index=False)
    summary = {"config": as_dict(cfg), "load_report": rep, "window_report": wrep}
    (out / "real_csv_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"[csv] {len(win_df)} windows -> {out}")
    return win_df, summary
