
"""Process raw engine waveforms into window-level feature data."""

from pathlib import Path

import numpy as np
import pandas as pd

from signal_features import extract_signal_features

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

CONDITIONS = [
    "normal",
    "misfire",
    "knock",
    "valve_train_wear",
    "overheating",
]

WINDOW_SECONDS = 0.5
EXPECTED_SAMPLE_RATE = 4000


def process_waveform_file(path):
    """Extract features from consecutive windows in one waveform CSV."""
    df = pd.read_csv(path)

    required = {
        "timestamp_s",
        "microphone_audio",
        "knock_sensor",
        "crank_pulse",
        "condition",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    timestamps = df["timestamp_s"].to_numpy(dtype=float)
    if len(timestamps) < 2:
        raise ValueError(f"{path.name} has too few samples.")

    sample_rate = 1.0 / np.median(np.diff(timestamps))
    if not np.isclose(sample_rate, EXPECTED_SAMPLE_RATE, rtol=0.02):
        print(f"Warning: {path.name} sample rate is {sample_rate:.1f} Hz")

    samples_per_window = int(round(WINDOW_SECONDS * sample_rate))
    rows = []

    for start in range(0, len(df) - samples_per_window + 1, samples_per_window):
        end = start + samples_per_window
        window = df.iloc[start:end]

        row = {
            "condition": str(window["condition"].iloc[0]),
            "source_file": path.name,
            "window_start_s": float(window["timestamp_s"].iloc[0]),
            "window_end_s": float(window["timestamp_s"].iloc[-1]),
            "sample_rate_hz": float(sample_rate),
        }

        row.update(
            extract_signal_features(
                window["microphone_audio"].to_numpy(),
                sample_rate,
                "mic",
            )
        )
        row.update(
            extract_signal_features(
                window["knock_sensor"].to_numpy(),
                sample_rate,
                "knock",
            )
        )

        crank = window["crank_pulse"].to_numpy(dtype=float)
        rising_edges = int(
            np.sum((crank[:-1] == 0) & (crank[1:] == 1))
        )
        row["crank_rising_edges"] = rising_edges
        row["crank_pulse_mean"] = float(np.mean(crank))
        row["crank_estimated_rpm"] = rising_edges / WINDOW_SECONDS * 60.0

        rows.append(row)

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for condition in CONDITIONS:
        path = RAW_DIR / f"{condition}_waveform.csv"
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        rows = process_waveform_file(path)
        all_rows.extend(rows)
        print(f"{condition}: extracted {len(rows)} windows")

    if not all_rows:
        raise FileNotFoundError(
            f"No waveform CSV files found in {RAW_DIR}. Run the simulator first."
        )

    result = pd.DataFrame(all_rows)
    output_path = OUTPUT_DIR / "window_features.csv"
    result.to_csv(output_path, index=False)

    print(f"\nSaved {len(result)} feature rows to:")
    print(output_path)
    print("\nRows per condition:")
    print(result.groupby("condition").size().to_string())


if __name__ == "__main__":
    main()
