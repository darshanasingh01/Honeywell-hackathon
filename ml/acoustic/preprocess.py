"""Audio loading, resampling, filtering, and fixed-length segmentation."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(path: str | Path, sample_rate: int = 22_050) -> np.ndarray:
    """Load mono float audio and resample it to a consistent rate."""
    audio, _ = librosa.load(Path(path), sr=sample_rate, mono=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Audio is empty or contains non-finite samples: {path}")
    return audio.astype(np.float32, copy=False)


def split_audio(audio: np.ndarray, sample_rate: int, seconds: float = 3.0) -> list[np.ndarray]:
    """Return non-overlapping fixed-duration chunks, dropping a short tail."""
    if sample_rate <= 0 or seconds <= 0:
        raise ValueError("sample_rate and seconds must be positive")
    chunk_size = int(round(sample_rate * seconds))
    return [audio[i : i + chunk_size] for i in range(0, len(audio) - chunk_size + 1, chunk_size)]
