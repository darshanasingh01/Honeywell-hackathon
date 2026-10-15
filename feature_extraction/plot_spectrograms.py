"""Plot knock-sensor spectrograms per condition for slides / reports.

Zooms to 2-13 kHz so the 5-10 kHz knock band is not drowned by the loud
low-frequency engine content, and sets contrast from the displayed band.

Run from the repo root:
    python -m feature_extraction.plot_spectrograms                 # normal vs knock
    python -m feature_extraction.plot_spectrograms knock misfire valve_train_wear
Figures are written to data/processed/ (git-ignored).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

FS = 25600
CHANNEL = "knock_sensor"
FREQ_LO_KHZ, FREQ_HI_KHZ = 2.0, 13.0
BAND_LO_KHZ, BAND_HI_KHZ = 5.0, 10.0
SLICE_START_S, SLICE_DUR_S = 1.0, 0.3


def _load_slice(condition, seed=0, t0=SLICE_START_S, dur=SLICE_DUR_S):
    path = RAW_DIR / f"{condition}_waveform_s{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m feature_extraction.generate_dataset` first."
        )
    df = pd.read_csv(path)
    t = df["timestamp_s"].to_numpy()
    mask = (t >= t0) & (t < t0 + dur)
    return df[CHANNEL].to_numpy()[mask]


def _spectrogram_db(signal, nperseg=256, noverlap=240):
    f, t, sxx = spectrogram(
        signal, fs=FS, nperseg=nperseg, noverlap=noverlap,
        window="hann", scaling="spectrum",
    )
    return f / 1000.0, t, 10 * np.log10(sxx + 1e-12)


def plot_conditions(conditions, seed=0, out_name=None):
    """Render one zoomed knock-sensor spectrogram panel per condition, shared contrast."""
    panels, vals = {}, []
    for cond in conditions:
        f, t, d = _spectrogram_db(_load_slice(cond, seed))
        band = (f >= FREQ_LO_KHZ) & (f <= FREQ_HI_KHZ)
        panels[cond] = (f[band], t, d[band])
        vals.append(d[band].ravel())
    vals = np.concatenate(vals)
    vmin, vmax = np.percentile(vals, 40), np.percentile(vals, 99.8)

    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.8), constrained_layout=True, squeeze=False)
    im = None
    for ax, cond in zip(axes[0], conditions):
        f, t, d = panels[cond]
        im = ax.pcolormesh(t, f, d, cmap="inferno", vmin=vmin, vmax=vmax, shading="gouraud")
        ax.axhline(BAND_LO_KHZ, color="cyan", lw=1.3, ls="--")
        ax.axhline(BAND_HI_KHZ, color="cyan", lw=1.3, ls="--")
        ax.text(t[0] + 0.004, BAND_HI_KHZ + 0.4, f"{BAND_LO_KHZ:g}-{BAND_HI_KHZ:g} kHz knock band",
                color="cyan", fontsize=9.5, fontweight="bold")
        ax.set_title(cond.upper(), fontsize=13, fontweight="bold")
        ax.set_xlabel("time (s)", fontsize=10)
        ax.set_ylabel("frequency (kHz)", fontsize=10)
        ax.set_ylim(FREQ_LO_KHZ, FREQ_HI_KHZ)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9, pad=0.015)
    cb.set_label("power (dB)", fontsize=10)
    fig.suptitle(
        f"Knock-sensor spectrogram ({FREQ_LO_KHZ:g}-{FREQ_HI_KHZ:g} kHz) - "
        "knock shows bright bursts in the 5-10 kHz band",
        fontsize=12, fontweight="bold",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = out_name or ("spectrogram_" + "_vs_".join(conditions) + ".png")
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    return out_path


def main():
    conditions = sys.argv[1:] or ["normal", "knock"]
    out = plot_conditions(conditions)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
