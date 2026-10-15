"""Compact, interpretable spectral features for engine audio segments."""

from __future__ import annotations

import numpy as np
import librosa


def extract_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract robust segment statistics from MFCC, spectral, and energy features."""
    if audio.ndim != 1 or len(audio) < 2048:
        raise ValueError("Expected mono audio with at least 2048 samples")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite values")

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, n_fft=2048, hop_length=512)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=2048, hop_length=512)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, n_fft=2048, hop_length=512)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=2048, hop_length=512)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=2048, hop_length=512)

    matrix = np.vstack((mfcc, centroid, bandwidth, rolloff, rms, zcr))
    # Per-feature mean and standard deviation over time preserve the overall
    # spectral signature while reducing sensitivity to exact segment alignment.
    return np.concatenate((matrix.mean(axis=1), matrix.std(axis=1))).astype(np.float32)
