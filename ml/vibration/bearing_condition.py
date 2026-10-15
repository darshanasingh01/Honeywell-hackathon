"""Recording-level, sample-domain features for the NUST bearing-condition data.

No sampling rate or uniform sampling assumption is used here. Each CSV is one
example; its Time column is inspected for quality, not used to make frequency
features or fixed-duration windows.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


HEALTHY_LABEL = "healthy_bearing"
FAULTY_LABEL = "faulty_bearing"
LABELS = (HEALTHY_LABEL, FAULTY_LABEL)
CHANNEL = "Channel 3"
TIME_COLUMN = "Time"
CHANNEL_COLUMNS = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
MODEL_VERSION = "engineguard-vibration-nust-bearing-condition-v1"


@dataclass(frozen=True)
class BearingRecording:
    path: Path
    label: str
    recording_date: str | None
    samples: np.ndarray
    timestamp_quality: dict[str, int | bool | float | None]
    removed_initial_duplicate: bool
    sha256: str


def _label_for_path(path: Path) -> str:
    parts = {part.casefold() for part in path.parts}
    if "healthy bearings dataset" in parts:
        return HEALTHY_LABEL
    if "faulty bearings dataset" in parts:
        return FAULTY_LABEL
    raise ValueError(f"Could not identify a bearing-condition label from path: {path}")


def _date_for_name(path: Path) -> str | None:
    match = re.search(r"(20\d{2}[A-Za-z]{3}\d{2})", path.name)
    return match.group(1) if match else None


def _timestamp_quality(times: np.ndarray) -> dict[str, int | bool | float | None]:
    diffs = np.diff(times)
    positive = diffs[diffs > 0]
    median_step = float(np.median(positive)) if positive.size else None
    large_gap_count = int(np.sum(positive > max(0.01, 5.0 * median_step))) if median_step else 0
    uniform = bool(positive.size > 0 and np.allclose(positive, np.median(positive), rtol=0.01, atol=1e-9)
                   and not np.any(diffs <= 0))
    return {
        "row_count": int(times.size),
        "timestamp_repeats": int(np.sum(diffs == 0)),
        "timestamp_nonmonotonic_steps": int(np.sum(diffs < 0)),
        "large_gap_count_over_10ms_or_5x_median": large_gap_count,
        "timestamp_uniform": uniform,
        "time_start_seconds": float(times[0]) if times.size else None,
        "time_end_seconds": float(times[-1]) if times.size else None,
        "sample_rate_verified": False,
    }


def load_recording(path: str | Path, *, require_dataset_label: bool = True) -> BearingRecording:
    """Read a NUST-style CSV, validate Channel 3, and remove one exact first repeat."""
    path = Path(path)
    frame = pd.read_csv(path)
    missing = {TIME_COLUMN, *CHANNEL_COLUMNS} - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    channel = pd.to_numeric(frame[CHANNEL], errors="coerce").to_numpy(dtype=np.float64)
    if channel.size < 16 or not np.isfinite(channel).all():
        raise ValueError("Channel 3 must contain at least 16 finite numeric samples")
    times = pd.to_numeric(frame[TIME_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    if times.size != channel.size or not np.isfinite(times).all():
        raise ValueError("Time must contain a finite timestamp for every Channel 3 sample")

    removed = False
    if len(frame) >= 2:
        initial_columns = [TIME_COLUMN, *CHANNEL_COLUMNS]
        if frame.loc[0, initial_columns].equals(frame.loc[1, initial_columns]):
            frame = frame.iloc[1:].reset_index(drop=True)
            channel = channel[1:]
            times = times[1:]
            removed = True
    if channel.size < 16:
        raise ValueError("Fewer than 16 samples remain after duplicate removal")
    label = _label_for_path(path) if require_dataset_label else "input_recording"
    return BearingRecording(
        path=path,
        label=label,
        recording_date=_date_for_name(path),
        samples=channel,
        timestamp_quality=_timestamp_quality(times),
        removed_initial_duplicate=removed,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def discover_recordings(root: str | Path) -> tuple[list[BearingRecording], dict[str, int]]:
    """Discover labelled CSVs and deduplicate by exact file bytes before splitting."""
    root = Path(root)
    paths = sorted(root.rglob("*.csv"))
    if not paths:
        raise ValueError(f"No CSV recordings found under {root}")
    by_hash: dict[str, BearingRecording] = {}
    duplicate_files = 0
    for path in paths:
        recording = load_recording(path)
        existing = by_hash.get(recording.sha256)
        if existing is not None:
            if existing.label != recording.label:
                raise ValueError(f"Identical recording content has conflicting labels: {existing.path}, {path}")
            duplicate_files += 1
            continue
        by_hash[recording.sha256] = recording
    recordings = list(by_hash.values())
    return recordings, {"discovered_csv_files": len(paths), "deduplicated_files": duplicate_files}


def extract_time_domain_features(samples: np.ndarray) -> dict[str, float]:
    """Extract one vector of exported-sample amplitude statistics; no Fs required."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size < 16 or not np.isfinite(x).all():
        raise ValueError("Expected at least 16 finite Channel 3 samples")
    centered = x - np.mean(x)
    std = float(np.std(centered))
    abs_peak = float(np.max(np.abs(centered)))
    rms = float(np.sqrt(np.mean(centered * centered)))
    median = float(np.median(centered))
    mad = float(np.median(np.abs(centered - median)))
    near_zero_scale = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(x)))) * 32.0
    if std <= near_zero_scale:
        skewness = 0.0
        excess_kurtosis = 0.0
    else:
        skewness = float(skew(centered, bias=False))
        excess_kurtosis = float(kurtosis(centered, fisher=True, bias=False))
    features = {
        "centered_rms_sample_units": rms,
        "standard_deviation_sample_units": std,
        "peak_to_peak_sample_units": float(np.ptp(x)),
        "absolute_peak_centered_sample_units": abs_peak,
        "crest_factor": abs_peak / rms if rms > np.finfo(np.float64).eps else 0.0,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "median_absolute_deviation_sample_units": mad,
        "q05_centered_sample_units": float(np.quantile(centered, 0.05)),
        "q25_centered_sample_units": float(np.quantile(centered, 0.25)),
        "q50_centered_sample_units": float(np.quantile(centered, 0.50)),
        "q75_centered_sample_units": float(np.quantile(centered, 0.75)),
        "q95_centered_sample_units": float(np.quantile(centered, 0.95)),
    }
    if not np.isfinite(list(features.values())).all():
        raise ValueError("Feature extraction produced a non-finite value")
    return features


FEATURE_NAMES = list(extract_time_domain_features(np.linspace(-1.0, 1.0, 16)).keys())
