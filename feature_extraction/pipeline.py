"""Window-level feature + spectrogram pipeline (25.6 kHz).

Reads raw runs, slices overlapped windows, and for each window produces a
74-feature row plus a 2-channel spectrogram. Telemetry is taken ONLY from the
decoded CAN log (never the raw telemetry CSV), matching the live Person D path.

Three row-aligned outputs are written:
  data/processed/features.csv          (meta cols + 74 feature cols)
  data/processed/spectrograms.npy      (N, 2, F, T) float32
  data/processed/spectrogram_meta.json (per-row meta + global settings)

Run from repo root: ``python -m feature_extraction.pipeline``
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import stft

from simulator.engine_simulator import CONDITIONS
from feature_extraction.can_codec import read_candump_log, can_to_telemetry
from feature_extraction.signal_features import (
    extract_fast_channel_features,
    extract_crank_features,
    extract_telemetry_features,
    crank_gated_knock,
    normalize_spectrogram_db,
    TELEMETRY_SIGNALS,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

FS_EXPECTED = 25600
WINDOW_SECONDS = 0.5
HOP_SECONDS = 0.25
NPERSEG = 128
SPEC_OVERLAP = 0.5

SEEDS = (0, 1, 2, 3, 4, 5)

META_COLUMNS = ["run_id", "window_id", "label", "t_start", "t_end"]

# ---- Exact 74-feature column order (the training contract) ----
_FAST_KEYS = [
    "std", "rms", "peak", "crest_factor", "kurtosis", "zero_crossing_rate",
    "dominant_frequency_hz", "spectral_centroid_hz",
    "low_0_1k_energy", "low_0_1k_ratio",
    "mid_1k_5k_energy", "mid_1k_5k_ratio",
    "knock_5k_10k_energy", "knock_5k_10k_ratio",
    "env_mean", "env_std", "env_peak", "env_peak_to_mean", "env_repetition_hz",
]
_CRANK_KEYS = ["crank_rpm", "crank_rpm_std", "crank_interval_cv"]
_TEL_STATS = ["mean", "std", "min", "max", "slope"]
_GATED_KEYS = [
    "knock_cycle_intensity_mean",
    "knock_cycle_intensity_max",
    "knock_cycle_pct_knocking",
]

FEATURE_COLUMNS = (
    [f"mic_{k}" for k in _FAST_KEYS]
    + [f"knock_{k}" for k in _FAST_KEYS]
    + list(_CRANK_KEYS)
    + [f"tel_{sig}_{stat}" for sig in TELEMETRY_SIGNALS for stat in _TEL_STATS]
    + list(_GATED_KEYS)
)


def iter_windows(n_samples, fs, window_s=WINDOW_SECONDS, hop_s=HOP_SECONDS):
    """Yield (start, end) sample-index pairs for overlapped windows."""
    win = int(round(window_s * fs))
    hop = int(round(hop_s * fs))
    if win <= 0 or hop <= 0:
        return
    start = 0
    while start + win <= n_samples:
        yield start, start + win
        start += hop


def compute_spectrogram(mic, knock, fs, nperseg=NPERSEG):
    """2-channel (mic, knock) dB spectrogram, per-window normalized, float32."""
    noverlap = int(round(nperseg * SPEC_OVERLAP))
    chans = []
    for sig in (mic, knock):
        _, _, Z = stft(np.asarray(sig, dtype=float), fs=fs,
                       nperseg=nperseg, noverlap=noverlap, boundary=None)
        mag = np.abs(Z)
        spec_db = 20.0 * np.log10(mag + 1e-10)
        spec_db = normalize_spectrogram_db(spec_db)
        chans.append(spec_db.astype(np.float32))
    return np.stack(chans, axis=0).astype(np.float32)


def features_for_window(mic, knock, crank, telemetry_rows, fs=FS_EXPECTED):
    """Shared core: one window -> (feature_row dict, spectrogram ndarray).

    Used for both the training dataset and the Person D live path.
    """
    row = {}
    row.update(extract_fast_channel_features(mic, fs, "mic"))
    row.update(extract_fast_channel_features(knock, fs, "knock"))
    row.update(extract_crank_features(crank, fs))
    row.update(extract_telemetry_features(telemetry_rows))
    row.update(crank_gated_knock(knock, crank, fs))

    spec = compute_spectrogram(mic, knock, fs)
    return row, spec


def process_run(waveform_df, telemetry_df, run_id, label, fs):
    """Window one run into aligned (feature rows, spectrograms)."""
    mic = waveform_df["microphone_audio"].to_numpy(dtype=float)
    knock = waveform_df["knock_sensor"].to_numpy(dtype=float)
    crank = waveform_df["crank_pulse"].to_numpy(dtype=float)
    ts = waveform_df["timestamp_s"].to_numpy(dtype=float)
    has_cond = "condition" in waveform_df.columns
    cond = waveform_df["condition"].to_numpy() if has_cond else None

    rows = []
    specs = []
    window_id = 0
    for start, end in iter_windows(len(waveform_df), fs):
        # Skip windows that straddle two labels.
        if cond is not None:
            seg_labels = pd.unique(cond[start:end])
            if len(seg_labels) != 1:
                continue
            win_label = str(seg_labels[0])
        else:
            win_label = label

        t_start = float(ts[start])
        t_end = float(ts[end - 1])

        # Telemetry rows within [t_start, t_end] from the DECODED CAN log.
        tel_win = telemetry_df[
            (telemetry_df["timestamp_s"] >= t_start)
            & (telemetry_df["timestamp_s"] <= t_end)
        ]

        feat, spec = features_for_window(
            mic[start:end], knock[start:end], crank[start:end], tel_win, fs
        )
        feat_row = {
            "run_id": run_id,
            "window_id": window_id,
            "label": win_label,
            "t_start": t_start,
            "t_end": t_end,
        }
        feat_row.update(feat)
        rows.append(feat_row)
        specs.append(spec)
        window_id += 1

    return rows, specs


def _derive_fs(ts):
    dt = np.median(np.diff(ts))
    return 1.0 / dt if dt > 0 else 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_specs = []

    for cond in CONDITIONS:
        for seed in SEEDS:
            wpath = RAW_DIR / f"{cond}_waveform_s{seed}.csv"
            cpath = RAW_DIR / f"{cond}_can_s{seed}.log"
            if not wpath.exists() or not cpath.exists():
                print(f"Skipping missing run: {cond} seed={seed}")
                continue

            waveform = pd.read_csv(wpath)
            ts = waveform["timestamp_s"].to_numpy(dtype=float)
            fs = _derive_fs(ts)
            if not np.isclose(fs, FS_EXPECTED, rtol=0.01):
                raise ValueError(
                    f"{wpath.name}: derived fs={fs:.1f} Hz != {FS_EXPECTED} Hz"
                )
            fs = FS_EXPECTED

            # TELEMETRY ONLY FROM DECODED CAN LOG.
            can_log = read_candump_log(cpath)
            telemetry = can_to_telemetry(can_log)

            run_id = f"{cond}_s{seed}"
            rows, specs = process_run(waveform, telemetry, run_id, cond, fs)
            all_rows.extend(rows)
            all_specs.extend(specs)
            print(f"{run_id:22s} windows={len(rows)}")

    if not all_rows:
        raise FileNotFoundError(
            f"No runs found in {RAW_DIR}. Run generate_dataset first."
        )

    # Enforce the exact column contract/order.
    features = pd.DataFrame(all_rows)
    expected_cols = META_COLUMNS + FEATURE_COLUMNS
    missing = set(expected_cols) - set(features.columns)
    extra = set(features.columns) - set(expected_cols)
    if missing or extra:
        raise ValueError(f"Feature column mismatch. missing={sorted(missing)} "
                         f"extra={sorted(extra)}")
    features = features[expected_cols]

    specs_arr = np.stack(all_specs, axis=0).astype(np.float32)

    meta = {
        "fs": FS_EXPECTED,
        "nperseg": NPERSEG,
        "spec_overlap": SPEC_OVERLAP,
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "channels": ["mic", "knock"],
        "spectrogram_shape": list(specs_arr.shape[1:]),
        "feature_columns": FEATURE_COLUMNS,
        "meta_columns": META_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "rows": [
            {
                "run_id": r["run_id"],
                "window_id": int(r["window_id"]),
                "label": r["label"],
                "t_start": float(r["t_start"]),
                "t_end": float(r["t_end"]),
            }
            for r in all_rows
        ],
    }

    assert len(features) == specs_arr.shape[0] == len(meta["rows"]), \
        "row-count misalignment"

    features.to_csv(OUTPUT_DIR / "features.csv", index=False)
    np.save(OUTPUT_DIR / "spectrograms.npy", specs_arr)
    with open(OUTPUT_DIR / "spectrogram_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nfeatures.csv      : {features.shape[0]} rows x {features.shape[1]} cols "
          f"({len(FEATURE_COLUMNS)} features + {len(META_COLUMNS)} meta)")
    print(f"spectrograms.npy  : {specs_arr.shape} {specs_arr.dtype}")
    print(f"spectrogram_meta  : {len(meta['rows'])} rows")
    print("\nwindows per label:")
    print(features.groupby("label").size().to_string())


if __name__ == "__main__":
    main()
