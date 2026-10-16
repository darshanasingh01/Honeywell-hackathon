"""Export simulator-reconstructed, V2-window-aligned fast-sensor STFTs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import stft

from .producer_v2 import FS, SOURCE_COMMIT, TELEMETRY_HZ, approved_sources
from .schema_v2 import FEATURE_COLUMNS, MANIFEST
from .verify_producer_v2 import verify_directory

DEFAULT_DATASET = Path("ml/datasets/synthetic_conditions_v2/training")
DEFAULT_OUTPUT = Path("ml/synthetic_conditions/fast_spectrograms_v2_complete_retry")
CONDITIONS = tuple(MANIFEST["class_labels"])
WINDOW_ID = 0
N_PER_SEG = 1024
N_OVERLAP = 768
N_FFT = 1024
STFT_WINDOW = "hann"
CHANNELS = {
    "microphone": ("microphone_audio", "mic"),
    "knock": ("knock_sensor", "knock"),
    "accelerometer_x": ("accel_x", "accel_x"),
    "accelerometer_y": ("accel_y", "accel_y"),
    "accelerometer_z": ("accel_z", "accel_z"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_match(row, waveform, dsp, start_index: int, end_index: int) -> dict[str, float]:
    errors: dict[str, float] = {}
    for _name, (column, prefix) in CHANNELS.items():
        derived = dsp.extract_fast_channel_features(
            waveform[column].to_numpy()[start_index:end_index], FS, prefix)
        for name, value in derived.items():
            if name not in FEATURE_COLUMNS:
                continue
            error = abs(float(value) - float(row[name]))
            errors[name] = error
            if error > 1e-8:
                raise ValueError(f"reconstructed feature mismatch for {name}: {error}")
    return errors


def _spectrogram(signal: np.ndarray, start_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, values = stft(
        signal, fs=FS, window=STFT_WINDOW, nperseg=N_PER_SEG,
        noverlap=N_OVERLAP, nfft=N_FFT, boundary=None, padded=False,
        return_onesided=True)
    absolute_times = times + start_s
    return frequencies.astype(np.float32), absolute_times.astype(np.float32), np.abs(values).astype(np.float32)


def export_fast_spectrograms(dataset=DEFAULT_DATASET, output=DEFAULT_OUTPUT):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing export directory: {output}")
    verification = verify_directory(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((dataset / "run_evidence.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(dataset / "features.csv", float_precision="round_trip")
    if manifest["configuration"]["waveform_hz"] != FS:
        raise ValueError("dataset waveform rate does not match producer contract")
    if set(frame.label) != set(CONDITIONS):
        raise ValueError("dataset does not contain all five V2 conditions")
    by_condition = {run["condition"]: run for run in evidence if run["seed"] == 3200}
    if set(by_condition) != set(CONDITIONS):
        raise ValueError("expected seed-3200 evidence for every V2 condition")
    simulator, dsp, _, _ = approved_sources()
    duration_s = float(manifest["configuration"]["duration_s"])
    output.mkdir(parents=True, exist_ok=False)
    exports = {}
    for condition in CONDITIONS:
        run = by_condition[condition]
        run_id = str(run["run_id"])
        rows = frame[(frame.run_id == run_id) & (frame.window_id == WINDOW_ID)]
        if len(rows) != 1:
            raise ValueError(f"expected exactly one V2 window for {run_id}")
        row = rows.iloc[0]
        start_s, end_s = float(row.t_start), float(row.t_end)
        start_index, end_index = round(start_s * FS), round(end_s * FS)
        if end_index <= start_index or end_index > round(duration_s * FS):
            raise ValueError(f"invalid waveform window for {run_id}/{WINDOW_ID}")
        source, waveform = simulator.simulate_engine(condition, duration_s, TELEMETRY_HZ, FS, int(run["seed"]))
        if len(waveform) != round(duration_s * FS):
            raise ValueError(f"unexpected reconstructed waveform length for {run_id}")
        timestamps = waveform.timestamp_s.to_numpy()
        expected = np.arange(len(waveform), dtype=float) / FS
        if not np.allclose(timestamps, expected, rtol=0.0, atol=1e-12):
            raise ValueError("reconstructed waveform clock is not sample-aligned")
        feature_errors = _feature_match(row, waveform, dsp, start_index, end_index)
        arrays: dict[str, np.ndarray] = {}
        frequency_axis = time_axis = None
        shapes = {}
        for name, (column, _prefix) in CHANNELS.items():
            frequencies, times, magnitude = _spectrogram(
                waveform[column].to_numpy()[start_index:end_index], start_s)
            if frequency_axis is None:
                frequency_axis, time_axis = frequencies, times
            elif not (np.array_equal(frequency_axis, frequencies) and np.array_equal(time_axis, times)):
                raise ValueError("channel STFT axes are not synchronized")
            arrays[name + "_magnitude"] = magnitude
            shapes[name] = list(magnitude.shape)
        if frequency_axis is None or time_axis is None:
            raise ValueError("no fast channels available")
        if not (np.all(np.isfinite(frequency_axis)) and np.all(np.isfinite(time_axis))
                and all(np.all(np.isfinite(value)) for value in arrays.values())):
            raise ValueError("non-finite spectrogram output")
        metadata = {
            "format_version": "engineguard-fast-spectrogram/1",
            "data_kind": "synthetic_simulated_engine_conditions",
            "real_engine_measurements": False,
            "condition_label": condition, "run_id": run_id, "window_id": WINDOW_ID,
            "window": {"t_start": start_s, "t_end": end_s, "semantics": "half-open [t_start,t_end)"},
            "channels": {name: {"source_column": column, "sample_rate_hz": FS,
                                  "units": "synthetic waveform amplitude"}
                         for name, (column, _prefix) in CHANNELS.items()},
            "stft": {"window": STFT_WINDOW, "nperseg": N_PER_SEG,
                     "noverlap": N_OVERLAP, "nfft": N_FFT,
                     "boundary": None, "padded": False,
                     "frequency_axis_units": "Hz", "time_axis_units": "run_clock_seconds"},
            "array_shapes": shapes,
            "alignment": {"sample_start": start_index, "sample_end_exclusive": end_index,
                          "sample_count": end_index - start_index,
                          "feature_match_max_abs_error": max(feature_errors.values()),
                          "feature_match_count": len(feature_errors),
                          "same_generated_run": True, "same_v2_window": True},
            "provenance": {"source_commit": SOURCE_COMMIT, "producer_version": manifest["producer_version"],
                           "dataset_schema_version": manifest["schema_version"],
                           "dataset_features_sha256": manifest["features_sha256"],
                           "seed": int(run["seed"]), "source_id": str(row.source_id)},
            "limitations": ["Synthetic simulator reconstruction; not a real-engine waveform.",
                            "Raw fast arrays were not retained in the dataset handoff; deterministic reconstruction was verified against all 95 fast waveform feature columns.",
                            "Spectrogram magnitudes are not additional model features."],
        }
        file_name = f"{condition}_window{WINDOW_ID}.npz"
        path = output / file_name
        np.savez_compressed(path, frequencies_hz=frequency_axis, times_s=time_axis,
                            metadata_json=np.asarray(json.dumps(metadata)), **arrays)
        exports[condition] = {"path": file_name, "sha256": _sha256(path),
                              "run_id": run_id, "window_id": WINDOW_ID,
                              "condition_label": condition, "array_shapes": shapes,
                              "feature_match_max_abs_error": max(feature_errors.values())}
    index = {"format_version": "engineguard-fast-spectrogram/1",
             "data_kind": "synthetic_simulated_engine_conditions",
             "real_engine_measurements": False, "source_dataset": str(dataset.as_posix()),
             "source_dataset_features_sha256": manifest["features_sha256"],
             "source_verification_status": verification["status"],
             "sample_rate_hz": FS, "stft": {"window": STFT_WINDOW, "nperseg": N_PER_SEG,
             "noverlap": N_OVERLAP, "nfft": N_FFT, "boundary": None, "padded": False},
             "channels": list(CHANNELS), "exports": exports,
             "limitations": ["Synthetic-only simulator-derived signatures.",
                             "No waveform was fabricated from engineered features."]}
    (output / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export_fast_spectrograms(args.dataset, args.output), indent=2))


if __name__ == "__main__":
    main()
