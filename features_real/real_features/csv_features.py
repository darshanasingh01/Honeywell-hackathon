"""Windowed features for the engine-signal CSV (EngineFaultDB_Final.csv).

The CSV is a time series with no timestamp or recording id, stored as one contiguous block per Fault class.
Neighbouring rows are near-copies (lag-1 autocorrelation ~0.99), so a random row split would leak.
Instead each class block is cut into contiguous chunks; whole chunks go to train / val / test, and rows near
every chunk edge are purged so no window sits next to one from another split.
"""
import re

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def clean_name(col):
    return re.sub(r"[^0-9a-zA-Z]+", "_", col).strip("_").lower()


def load_csv(path, label_col="Fault"):
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"column '{label_col}' not found; columns are {list(df.columns)}")
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)   # keeps first occurrence and the original order
    report = {"rows_in_file": n0, "exact_duplicate_rows_dropped": n0 - len(df)}
    if df.isna().any().any():
        raise ValueError("CSV contains missing values; clean them first")
    return df, report


def label_runs(labels):
    """Run id for each row = consecutive stretch with the same label."""
    s = pd.Series(labels)
    return (s != s.shift()).cumsum().to_numpy() - 1


def assign_chunk_splits(chunks, val_frac, test_frac, seed):
    """chunks: DataFrame with columns chunk_id, label. Stratified by label. Returns {chunk_id: split}."""
    rng = np.random.default_rng(seed)
    out = {}
    for lab, grp in chunks.groupby("label"):
        ids = grp["chunk_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        for i, cid in enumerate(ids):
            out[int(cid)] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return out


def build_csv_windows(df, cfg):
    """Returns (windows DataFrame, report dict)."""
    label = df[cfg.label_col].to_numpy()
    sig_cols = [c for c in df.columns if c != cfg.label_col]
    X = df[sig_cols].to_numpy(dtype=np.float64)
    runs = label_runs(label)
    n_runs = int(runs.max()) + 1
    n_classes = len(np.unique(label))

    # 1. contiguous chunks inside each run
    chunk_rows = []   # (chunk_id, label, run_id, start, end)  end exclusive
    for r in range(n_runs):
        idx = np.flatnonzero(runs == r)
        start, stop = int(idx[0]), int(idx[-1]) + 1
        for s in range(start, stop, cfg.chunk_rows):
            chunk_rows.append((len(chunk_rows), int(label[start]), r, s, min(s + cfg.chunk_rows, stop)))
    chunks = pd.DataFrame(chunk_rows, columns=["chunk_id", "label", "run_id", "start", "end"])
    split_of = assign_chunk_splits(chunks, cfg.val_frac, cfg.test_frac, cfg.seed)

    # 2. windows strictly inside each chunk after purging its edges
    t = np.arange(cfg.win_rows, dtype=np.float64)
    t = t - t.mean()
    denom = (t ** 2).sum()
    rows = []
    for ch in chunks.itertuples(index=False):
        a, b = ch.start + cfg.purge_rows, ch.end - cfg.purge_rows
        if b - a < cfg.win_rows:
            continue
        seg = X[a:b]
        win = sliding_window_view(seg, cfg.win_rows, axis=0)[::cfg.hop_rows]   # (n, n_sig, win_rows)
        mean = win.mean(axis=2)
        std = win.std(axis=2)
        mn, mx = win.min(axis=2), win.max(axis=2)
        slope = ((win - mean[:, :, None]) * t[None, None, :]).sum(axis=2) / denom
        n = win.shape[0]
        starts = a + np.arange(n) * cfg.hop_rows
        base = {
            "chunk_id": np.full(n, ch.chunk_id), "run_id": np.full(n, ch.run_id),
            "split": np.full(n, split_of[ch.chunk_id]),
            "row_start": starts, "row_end": starts + cfg.win_rows,
            cfg.label_col.lower(): np.full(n, ch.label),
        }
        feats = {}
        for j, c in enumerate(sig_cols):
            nm = clean_name(c)
            feats[f"{nm}_mean"] = mean[:, j]
            feats[f"{nm}_std"] = std[:, j]
            feats[f"{nm}_min"] = mn[:, j]
            feats[f"{nm}_max"] = mx[:, j]
            feats[f"{nm}_slope"] = slope[:, j]
        rows.append(pd.DataFrame({**base, **feats}))
    if not rows:
        hint = ("the rows look interleaved / not in time order (%d runs for %d classes); the file must keep its "
                "original order" % (n_runs, n_classes)) if n_runs > n_classes else \
               "reduce win_rows / purge_rows / chunk_rows"
        raise ValueError("no CSV windows produced: " + hint)
    win_df = pd.concat(rows, ignore_index=True)
    win_df.insert(0, "window_id", [f"csv_w{i:05d}" for i in range(len(win_df))])
    report = {
        "n_classes": n_classes,
        "n_runs": n_runs,
        "rows_in_contiguous_runs_note": "contiguous" if n_runs == n_classes else
            f"WARNING: {n_runs} runs for {n_classes} classes - rows look interleaved, window features may mix classes",
        "n_chunks": len(chunks),
        "n_windows": len(win_df),
        "windows_per_split": win_df["split"].value_counts().to_dict(),
        "windows_per_class_split": {f"{k[0]}|{k[1]}": int(v) for k, v in
                                    win_df.groupby([cfg.label_col.lower(), "split"]).size().items()},
    }
    return win_df, report
