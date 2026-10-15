"""Validate synthetic engine datasets and create diagnostic plots."""
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


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for condition in CONDITIONS:
        telemetry = pd.read_csv(DATA_DIR / f"{condition}_telemetry.csv")
        waveform = pd.read_csv(DATA_DIR / f"{condition}_waveform.csv")

        for name, frame in [("telemetry", telemetry), ("waveform", waveform)]:
            if frame.empty or frame.isnull().any().any():
                raise ValueError(f"{condition}: empty data or missing values in {name}")
            if frame["condition"].nunique() != 1:
                raise ValueError(f"{condition}: inconsistent labels in {name}")
            if frame["condition"].iloc[0] != condition:
                raise ValueError(f"{condition}: incorrect label in {name}")
            
            timestamps = frame["timestamp_s"].to_numpy(dtype=float)

            if not frame["timestamp_s"].is_monotonic_increasing:
                raise ValueError(f"{condition}: timestamps not increasing in {name}")

            expected_rate = 10 if name == "telemetry" else 4000
            expected_interval = 1.0 / expected_rate

            if len(timestamps) > 1:
                intervals = np.diff(timestamps)
                if not np.allclose(
                    intervals, expected_interval, rtol=1e-7, atol=1e-10
                ):
                    raise ValueError(
                        f"{condition}: irregular sampling in {name}; "
                        f"expected interval {expected_interval} s"
                    )


        rows.append({
            "condition": condition,
            "telemetry_rows": len(telemetry),
            "waveform_rows": len(waveform),
            "rpm_mean": telemetry["engine_rpm"].mean(),
            "coolant_temp_mean_c": telemetry["coolant_temp_c"].mean(),
            "coolant_temp_final_c": telemetry["coolant_temp_c"].iloc[-1],
            "ignition_timing_mean_deg": telemetry["ignition_timing_deg"].mean(),
            "audio_rms": rms(waveform["microphone_audio"]),
            "knock_sensor_rms": rms(waveform["knock_sensor"]),
            "accel_x_rms": rms(waveform["accel_x"]),
            "accel_y_rms": rms(waveform["accel_y"]),
            "accel_z_rms": rms(waveform["accel_z"]),
        })

        sample = waveform[waveform["timestamp_s"] <= 0.25]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(sample["timestamp_s"], sample["accel_x"], label="X")
        ax.plot(sample["timestamp_s"], sample["accel_y"], label="Y")
        ax.plot(sample["timestamp_s"], sample["accel_z"], label="Z")
        ax.set(title=f"Synthetic accelerometer: {condition}",
               xlabel="Time (s)", ylabel="Amplitude (arbitrary units)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{condition}_accelerometer.png", dpi=140)
        plt.close(fig)

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_DIR / "signal_validation.csv", index=False)

    for metric, filename, title in [
        ("ignition_timing_mean_deg", "ignition_timing_comparison.png",
         "Mean synthetic ignition timing"),
        ("accel_x_rms", "accelerometer_rms_comparison.png",
         "Accelerometer X-axis RMS"),
        ("knock_sensor_rms", "knock_sensor_rms_comparison.png",
         "Knock sensor RMS"),
    ]:
        ax = report.plot(x="condition", y=metric, kind="bar",
                         figsize=(9, 4), legend=False, rot=20, title=title)
        ax.set_ylabel(metric)
        ax.set_xlabel("Condition")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / filename, dpi=150)
        plt.close()

    print("Validation checks passed for all five conditions.")
    print(report.round(4).to_string(index=False))
    print(f"\nMetrics: {REPORT_DIR / 'signal_validation.csv'}")
    print(f"Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
