"""Strict, read-only access to the ``features_real/real_outputs`` exports.

This module deliberately does not join audio and CSV data, infer timestamps,
fill features, or invoke a trained classifier. All labels and identifiers are
returned exactly as stored by the producer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class RealDataAdapterError(ValueError):
    """Raised when an export is unavailable or violates its source contract."""


_AUDIO_MANIFEST_COLUMNS = {"recording_key", "rec_num", "fuel", "label", "path", "split"}
_AUDIO_WINDOW_COLUMNS = {
    "window_id", "recording_key", "rec_num", "fuel", "label", "split",
    "start_s", "end_s", "padded", "clipped",
}
_CSV_COLUMNS = {"window_id", "chunk_id", "run_id", "split", "row_start", "row_end", "fault"}
_SPLITS = {"train", "val", "test"}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


class RealDataAdapter:
    """Read teammate exports from a configurable directory without mutation."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"real-data export directory not found: {self.root}")
        self._audio_manifest: pd.DataFrame | None = None
        self._audio_windows: pd.DataFrame | None = None
        self._csv_windows: pd.DataFrame | None = None
        self._spectrograms: np.ndarray | None = None
        self._spectrogram_ids: list[str] | None = None

    def _file(self, name: str) -> Path:
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"required real-data artifact missing: {path}")
        return path

    @staticmethod
    def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
        missing = required - set(frame.columns)
        if missing:
            raise RealDataAdapterError(f"{name} missing columns: {sorted(missing)}")

    def _load_audio_manifest(self) -> pd.DataFrame:
        if self._audio_manifest is None:
            frame = pd.read_csv(self._file("real_audio_manifest.csv"))
            self._require_columns(frame, _AUDIO_MANIFEST_COLUMNS, "real_audio_manifest.csv")
            if frame["recording_key"].duplicated().any():
                raise RealDataAdapterError("audio manifest recording_key values must be unique")
            if not set(frame["split"].dropna()).issubset(_SPLITS):
                raise RealDataAdapterError("audio manifest contains an invalid split")
            if frame[["recording_key", "fuel", "split"]].isna().any().any():
                raise RealDataAdapterError("audio manifest has missing identity or split values")
            self._audio_manifest = frame
        return self._audio_manifest

    def _load_audio_windows(self) -> pd.DataFrame:
        if self._audio_windows is None:
            frame = pd.read_csv(self._file("real_audio_windows.csv"))
            self._require_columns(frame, _AUDIO_WINDOW_COLUMNS, "real_audio_windows.csv")
            if frame["window_id"].duplicated().any():
                raise RealDataAdapterError("audio window_id values must be unique")
            if not set(frame["split"].dropna()).issubset(_SPLITS):
                raise RealDataAdapterError("audio windows contain an invalid split")
            manifest = self._load_audio_manifest().set_index("recording_key")
            unknown = set(frame["recording_key"]) - set(manifest.index)
            if unknown:
                raise RealDataAdapterError(f"audio windows reference unknown recordings: {sorted(unknown)}")
            expected = frame["recording_key"].map(manifest["split"]).to_numpy()
            if not np.array_equal(expected, frame["split"].to_numpy()):
                raise RealDataAdapterError("audio window split disagrees with recording manifest")
            if not np.isfinite(frame[["start_s", "end_s"]].to_numpy(dtype=float)).all():
                raise RealDataAdapterError("audio window boundaries must be finite")
            self._audio_windows = frame
        return self._audio_windows

    def _load_spectrograms(self) -> tuple[np.ndarray, list[str]]:
        if self._spectrograms is None or self._spectrogram_ids is None:
            try:
                with np.load(self._file("real_audio_spectrograms.npz"), allow_pickle=True) as data:
                    if not {"spectrograms", "window_id"}.issubset(data.files):
                        raise RealDataAdapterError("spectrogram NPZ requires spectrograms and window_id arrays")
                    specs = np.asarray(data["spectrograms"])
                    ids = [str(value) for value in np.asarray(data["window_id"]).tolist()]
            except (OSError, ValueError, EOFError) as exc:
                raise RealDataAdapterError(f"unable to read spectrogram NPZ: {exc}") from exc
            if specs.ndim != 3 or specs.shape[0] != len(ids):
                raise RealDataAdapterError("spectrogram array and window_id lengths do not align")
            if len(set(ids)) != len(ids):
                raise RealDataAdapterError("spectrogram window_id values must be unique")
            windows = self._load_audio_windows()
            if set(ids) != set(windows["window_id"].astype(str)):
                raise RealDataAdapterError("spectrogram IDs do not match audio window CSV")
            self._spectrograms, self._spectrogram_ids = specs, ids
        return self._spectrograms, self._spectrogram_ids

    def _load_csv_windows(self) -> pd.DataFrame:
        if self._csv_windows is None:
            frame = pd.read_csv(self._file("real_csv_windows.csv"))
            self._require_columns(frame, _CSV_COLUMNS, "real_csv_windows.csv")
            if frame["window_id"].duplicated().any():
                raise RealDataAdapterError("CSV window_id values must be unique")
            if not set(frame["split"].dropna()).issubset(_SPLITS):
                raise RealDataAdapterError("CSV windows contain an invalid split")
            if not set(frame["fault"].dropna()).issubset({0, 1, 2, 3}):
                raise RealDataAdapterError("CSV fault codes must remain numeric values 0 through 3")
            if (frame["row_end"] <= frame["row_start"]).any():
                raise RealDataAdapterError("CSV row boundaries must be increasing")
            self._csv_windows = frame
        return self._csv_windows

    def list_audio_recordings(self) -> list[dict[str, Any]]:
        """Return recording metadata, including fuel and assigned split."""
        return _records(self._load_audio_manifest())

    def audio_windows(self, recording_key: str) -> list[dict[str, Any]]:
        """Return all feature rows for one recording, preserving source columns."""
        frame = self._load_audio_windows()
        if recording_key not in set(frame["recording_key"]):
            raise KeyError(f"audio recording not found: {recording_key}")
        return _records(frame[frame["recording_key"] == recording_key])

    def audio_spectrogram(self, window_id: str) -> dict[str, Any]:
        """Return the spectrogram aligned to a validated audio feature row."""
        specs, ids = self._load_spectrograms()
        if window_id not in set(ids):
            raise KeyError(f"audio spectrogram window not found: {window_id}")
        row = self._load_audio_windows()
        feature_row = row[row["window_id"].astype(str) == str(window_id)]
        index = ids.index(str(window_id))
        return {"window_id": str(window_id), "feature_row": _records(feature_row)[0],
                "spectrogram": specs[index].copy()}

    def list_csv_fault_codes(self) -> list[int]:
        """Return opaque numeric CSV codes; their meanings are intentionally unknown."""
        return sorted(int(value) for value in self._load_csv_windows()["fault"].dropna().unique())

    def list_csv_splits(self) -> list[str]:
        """Return the source-assigned CSV split names without recomputing them."""
        return sorted(str(value) for value in self._load_csv_windows()["split"].dropna().unique())

    def csv_windows(self, *, fault: int | None = None, split: str | None = None) -> list[dict[str, Any]]:
        """Return CSV feature windows with original row boundaries."""
        frame = self._load_csv_windows()
        if fault is not None:
            if fault not in {0, 1, 2, 3}:
                raise ValueError("fault must be one of numeric codes 0, 1, 2, 3")
            frame = frame[frame["fault"] == fault]
        if split is not None:
            if split not in _SPLITS:
                raise ValueError(f"split must be one of {sorted(_SPLITS)}")
            frame = frame[frame["split"] == split]
        return _records(frame)

    def metadata(self) -> dict[str, Any]:
        """Return producer summaries and adapter-level counts without inference."""
        audio_summary = json.loads(self._file("real_audio_summary.json").read_text(encoding="utf-8"))
        csv_summary = json.loads(self._file("real_csv_summary.json").read_text(encoding="utf-8"))
        manifest = self._load_audio_manifest()
        audio_windows = self._load_audio_windows()
        csv_windows = self._load_csv_windows()
        specs, _ = self._load_spectrograms()
        return {
            "data_kind": "real_recorded_unlabeled_audio_and_engine_csv",
            "audio": {"fault_label": "unknown", "fuel_labels": sorted(manifest["fuel"].unique().tolist()),
                       "recordings": int(len(manifest)), "windows": int(len(audio_windows)),
                       "spectrogram_shape": list(specs.shape)},
            "csv": {"fault_meaning": "unknown", "fault_codes": self.list_csv_fault_codes(),
                    "windows": int(len(csv_windows))},
            "producer_summaries": {"audio": audio_summary, "csv": csv_summary},
        }
