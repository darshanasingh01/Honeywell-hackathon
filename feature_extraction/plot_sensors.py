"""Plot the Person B sensor channels (3-axis accelerometer + ignition timing).

Companion to ``plot_spectrograms.py`` for the two sensors added on the
``features-sensors`` branch. Three figure kinds, all written to
``data/processed/`` (git-ignored):

  * accel spectrograms   - whitened 2-13 kHz spectrogram per condition for one
                           accelerometer axis (knock shows in accel_x 5-10 kHz,
                           valve wear in accel_y 1-5 kHz).
  * ignition timing      - ignition_timing (deg BTDC) vs time, one line per
                           condition (knock retards, misfire scatters).
  * sensor signatures    - band-ratio bars across the five fast channels
                           (mic, knock, accel_x/y/z) for one condition.

Run from the repo root:
    python -m feature_extraction.plot_sensors all                       # everything, 5 conditions
    python -m feature_extraction.plot_sensors accel knock normal        # accel spectrograms
    python -m feature_extraction.plot_sensors ignition                  # ignition trace, 5 conditions
    python -m feature_extraction.plot_sensors signature knock           # one signature figure
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from feature_extraction.signal_features import extract_band_features, FAST_BANDS

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

FS = 25600
CONDITIONS = ("normal", "misfire", "knock", "valve_train_wear", "overheating")
ACCEL_AXES = ("accel_x", "accel_y", "accel_z")
FAST_CHANNELS = ("microphone_audio", "knock_sensor", "accel_x", "accel_y", "accel_z")

FREQ_LO_KHZ, FREQ_HI_KHZ = 2.0, 13.0
KNOCK_LO_KHZ, KNOCK_HI_KHZ = 5.0, 10.0
MID_LO_KHZ, MID_HI_KHZ = 1.0, 5.0
SLICE_START_S, SLICE_DUR_S = 1.0, 0.3


def _waveform_path(condition, seed=0):
    path = RAW_DIR / f"{condition}_waveform_s{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m feature_extraction.generate_dataset` first."
        )
    return path


def _telemetry_path(condition, seed=0):
    path = RAW_DIR / f"{condition}_telemetry_s{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m feature_extraction.generate_dataset` first."
        )
    return path


def _load_slice(condition, channel, seed=0, t0=SLICE_START_S, dur=SLICE_DUR_S):
    df = pd.read_csv(_waveform_path(condition, seed))
    t = df["timestamp_s"].to_numpy()
    mask = (t >= t0) & (t < t0 + dur)
    return df[channel].to_numpy()[mask]


def _whitened_spectrogram_db(signal, nperseg=256, noverlap=240):
    """dB spectrogram whitened per frequency bin (row median removed)."""
    f, t, sxx = spectrogram(
        signal, fs=FS, nperseg=nperseg, noverlap=noverlap,
        window="hann", scaling="spectrum",
    )
    d = 10 * np.log10(sxx + 1e-12)
    d = d - np.median(d, axis=1, keepdims=True)  # whiten: flatten broadband floor
    return f / 1000.0, t, d


# ---------------------------------------------------------------------------
# 1. Accelerometer spectrograms
# ---------------------------------------------------------------------------
def plot_accel_spectrograms(conditions, axis="accel_x", seed=0, out_name=None):
    """Whitened 2-13 kHz spectrogram per condition for one accelerometer axis."""
    if axis not in ACCEL_AXES:
        raise ValueError(f"axis must be one of {ACCEL_AXES}")

    panels, vals = {}, []
    for cond in conditions:
        f, t, d = _whitened_spectrogram_db(_load_slice(cond, axis, seed))
        band = (f >= FREQ_LO_KHZ) & (f <= FREQ_HI_KHZ)
        panels[cond] = (f[band], t, d[band])
        vals.append(d[band].ravel())
    vals = np.concatenate(vals)
    vmin, vmax = np.percentile(vals, 50), np.percentile(vals, 99.8)

    # accel_x carries the knock band; accel_y the valve mid band; accel_z low.
    if axis == "accel_y":
        hi_lo, hi_hi, label = MID_LO_KHZ, MID_HI_KHZ, "1-5 kHz valve band"
    else:
        hi_lo, hi_hi, label = KNOCK_LO_KHZ, KNOCK_HI_KHZ, "5-10 kHz knock band"

    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.8),
                             constrained_layout=True, squeeze=False)
    im = None
    for ax, cond in zip(axes[0], conditions):
        f, t, d = panels[cond]
        im = ax.pcolormesh(t, f, d, cmap="inferno", vmin=vmin, vmax=vmax,
                           shading="gouraud")
        ax.axhline(hi_lo, color="cyan", lw=1.3, ls="--")
        ax.axhline(hi_hi, color="cyan", lw=1.3, ls="--")
        ax.text(t[0] + 0.004, hi_hi + 0.4, label,
                color="cyan", fontsize=9.5, fontweight="bold")
        ax.set_title(cond.upper(), fontsize=13, fontweight="bold")
        ax.set_xlabel("time (s)", fontsize=10)
        ax.set_ylabel("frequency (kHz)", fontsize=10)
        ax.set_ylim(FREQ_LO_KHZ, FREQ_HI_KHZ)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9, pad=0.015)
    cb.set_label("whitened power (dB)", fontsize=10)
    fig.suptitle(
        f"{axis} spectrogram ({FREQ_LO_KHZ:g}-{FREQ_HI_KHZ:g} kHz, whitened)",
        fontsize=12, fontweight="bold",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = out_name or f"{axis}_spectrogram_" + "_vs_".join(conditions) + ".png"
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2. Ignition timing trace
# ---------------------------------------------------------------------------
def plot_ignition_timing(conditions, seed=0, out_name=None):
    """ignition_timing (deg BTDC) vs time, one line per condition."""
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for cond in conditions:
        df = pd.read_csv(_telemetry_path(cond, seed))
        ax.plot(df["timestamp_s"], df["ignition_timing"], lw=1.8,
                marker="o", ms=3, label=cond)
    ax.set_xlabel("time (s)", fontsize=11)
    ax.set_ylabel("ignition timing (deg BTDC)", fontsize=11)
    ax.set_title("Ignition timing per condition - knock retards, misfire scatters",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = out_name or "ignition_timing_trace.png"
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3. Sensor signature (band-ratio bars across the five fast channels)
# ---------------------------------------------------------------------------
def plot_sensor_signatures(condition, seed=0, out_name=None):
    """Grouped band-ratio bars for the five fast channels, one condition."""
    df = pd.read_csv(_waveform_path(condition, seed))
    band_names = list(FAST_BANDS.keys())  # low_0_1k, mid_1k_5k, knock_5k_10k

    # channel -> [ratio per band]
    ratios = {}
    for ch in FAST_CHANNELS:
        feats = extract_band_features(df[ch].to_numpy(dtype=float), FS)
        ratios[ch] = [feats[f"{b}_ratio"] for b in band_names]

    chan_labels = ["mic", "knock", "accel_x", "accel_y", "accel_z"]
    x = np.arange(len(band_names))
    width = 0.16

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i, (ch, lab) in enumerate(zip(FAST_CHANNELS, chan_labels)):
        ax.bar(x + (i - 2) * width, ratios[ch], width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["0-1 kHz", "1-5 kHz", "5-10 kHz"], fontsize=10)
    ax.set_ylabel("band energy ratio", fontsize=11)
    ax.set_title(f"Sensor band signature - {condition.upper()}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, ncol=5, loc="upper center")
    ax.grid(True, axis="y", alpha=0.3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = out_name or f"sensor_signature_{condition}.png"
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    return out_path


def main():
    args = sys.argv[1:]
    mode = args[0] if args else "all"
    rest = args[1:]
    conditions = rest or list(CONDITIONS)

    saved = []
    if mode in ("accel", "all"):
        for axis in ACCEL_AXES:
            saved.append(plot_accel_spectrograms(conditions, axis=axis))
    if mode in ("ignition", "all"):
        saved.append(plot_ignition_timing(conditions))
    if mode in ("signature", "all"):
        for cond in conditions:
            saved.append(plot_sensor_signatures(cond))
    if mode not in ("accel", "ignition", "signature", "all"):
        print("usage: python -m feature_extraction.plot_sensors "
              "{accel|ignition|signature|all} [conditions...]")
        sys.exit(1)

    for p in saved:
        print(f"saved {p}")


if __name__ == "__main__":
    main()
