"""Basic quality checks and diagnostic plots for synthetic engine data."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")
REPORT_DIR = Path("data/processed")
PLOT_DIR = Path("docs/plots")

CONDITIONS = [
    "normal", "misfire", "knock", "valve_train_wear", "overheating"
]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for condition in CONDITIONS:
        # Per-seed dataset (seed 0) produced by feature_extraction.generate_dataset,
        # falling back to the legacy non-seed filenames from engine_simulator.main().
        telemetry_path = DATA_DIR / f"{condition}_telemetry_s0.csv"
        waveform_path = DATA_DIR / f"{condition}_waveform_s0.csv"
        if not telemetry_path.exists() or not waveform_path.exists():
            telemetry_path = DATA_DIR / f"{condition}_telemetry.csv"
            waveform_path = DATA_DIR / f"{condition}_waveform.csv"

        if not telemetry_path.exists() or not waveform_path.exists():
            raise FileNotFoundError(
                f"Missing CSV files for {condition}. "
                "Run python -m feature_extraction.generate_dataset first."
            )

        telemetry = pd.read_csv(telemetry_path)
        waveform = pd.read_csv(waveform_path)

        for name, frame in [("telemetry", telemetry), ("waveform", waveform)]:
            if frame.empty:
                raise ValueError(f"{condition}: {name} is empty")
            if frame.isnull().any().any():
                raise ValueError(f"{condition}: {name} contains missing values")
            if frame["condition"].nunique() != 1:
                raise ValueError(f"{condition}: inconsistent labels in {name}")
            if frame["condition"].iloc[0] != condition:
                raise ValueError(f"{condition}: unexpected label in {name}")
            if not frame["timestamp_s"].is_monotonic_increasing:
                raise ValueError(f"{condition}: timestamps are not increasing")

        audio = waveform["microphone_audio"].to_numpy()
        knock = waveform["knock_sensor"].to_numpy()
        temp = telemetry["coolant_temp_c"].to_numpy()

        rows.append({
            "condition": condition,
            "telemetry_rows": len(telemetry),
            "waveform_rows": len(waveform),
            "rpm_mean": telemetry["engine_rpm"].mean(),
            "coolant_temp_mean_c": temp.mean(),
            "coolant_temp_final_c": temp[-1],
            "audio_rms": np.sqrt(np.mean(audio ** 2)),
            "audio_peak_abs": np.max(np.abs(audio)),
            "knock_sensor_rms": np.sqrt(np.mean(knock ** 2)),
            "missing_telemetry_values": int(telemetry.isnull().sum().sum()),
            "missing_waveform_values": int(waveform.isnull().sum().sum()),
        })

        # Short time-domain comparison: first 0.25 seconds.
        sample = waveform[waveform["timestamp_s"] <= 0.25]
        plt.figure(figsize=(10, 3))
        plt.plot(sample["timestamp_s"], sample["microphone_audio"],
                 linewidth=0.7)
        plt.title(f"Synthetic microphone signal: {condition}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (arbitrary units)")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{condition}_audio.png", dpi=140)
        plt.close()

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_DIR / "signal_validation.csv", index=False)

    # Compare simple waveform metrics across conditions.
    ax = report.plot(
        x="condition",
        y=["audio_rms", "knock_sensor_rms"],
        kind="bar",
        figsize=(10, 5),
        rot=20,
        title="Synthetic waveform RMS by condition",
    )
    ax.set_ylabel("RMS amplitude (arbitrary units)")
    ax.set_xlabel("Simulated condition")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "condition_rms_comparison.png", dpi=150)
    plt.close()

    print("Validation checks passed for all five conditions.")
    print(report.round(4).to_string(index=False))
    print(f"\nMetrics: {REPORT_DIR / 'signal_validation.csv'}")
    print(f"Plots:   {PLOT_DIR}")


if __name__ == "__main__":
    main()
