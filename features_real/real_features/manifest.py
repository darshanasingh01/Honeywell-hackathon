"""Find the recordings, label them by fuel from the folder name, drop duplicate files, assign splits."""
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

FUELS = ("diesel", "petrol")
_REC_RE = re.compile(r"REC(\d+)", re.IGNORECASE)


def infer_fuel(path: Path, root: Path):
    """Return 'diesel' / 'petrol' from any folder name between root and the file, else None."""
    rel = path.relative_to(root)
    for part in rel.parts[:-1]:
        low = part.lower()
        for fuel in FUELS:
            if fuel in low:
                return fuel
    return None


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(audio_root):
    """Scan audio_root recursively for .wav files.

    Returns (manifest, report). The manifest has one row per UNIQUE recording (by file content).
    Files in folders whose name contains neither 'diesel' nor 'petrol' are skipped and counted.
    The same content under two different fuels raises ValueError (labels would conflict).
    """
    root = Path(audio_root)
    if not root.is_dir():
        raise FileNotFoundError(f"audio folder not found: {root}")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".wav" and "__MACOSX" not in p.parts
                   and not p.name.startswith("._"))
    report = {"wav_found": len(paths), "skipped_unlabelled_folder": 0, "duplicate_files_dropped": 0}
    rows, seen = [], {}
    for p in paths:
        fuel = infer_fuel(p, root)
        if fuel is None:
            report["skipped_unlabelled_folder"] += 1
            continue
        digest = file_sha1(p)
        if digest in seen:
            if seen[digest] != fuel:
                raise ValueError(f"identical audio appears under both fuels: {p}")
            report["duplicate_files_dropped"] += 1
            continue
        seen[digest] = fuel
        m = _REC_RE.search(p.stem)
        rows.append({
            "recording_key": p.stem.lower(),
            "rec_num": int(m.group(1)) if m else -1,
            "fuel": fuel,
            "label": FUELS.index(fuel),
            "path": str(p),
            "sha1": digest,
        })
    man = pd.DataFrame(rows)
    if man.empty:
        raise ValueError("no labelled .wav files found (folder names must contain 'diesel' or 'petrol')")
    if man["recording_key"].duplicated().any():
        dup = man.loc[man["recording_key"].duplicated(keep=False), "recording_key"].unique()[:5]
        raise ValueError(f"different audio shares a file name: {list(dup)}")
    man = man.sort_values(["rec_num", "recording_key"]).reset_index(drop=True)
    return man, report


def assign_recording_splits(manifest, val_frac=0.15, test_frac=0.15, seed=42):
    """Stratified-by-fuel split at the RECORDING level, so no recording is in two splits."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=manifest.index)
    for fuel, grp in manifest.groupby("fuel"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
    out = manifest.copy()
    out["split"] = split
    return out
