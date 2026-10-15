"""PASS/FAIL gate over Person B's three processed outputs.

Checklist (per the architect plan):
  1.  meta fs == 25600
  2.  row counts match: features == npy.shape[0] == meta rows
  3.  spectrograms ndim == 4 and channels == 2
  4.  no NaN/Inf in features or spectrograms
  5.  feature count == 74 and exact column set/order matches the contract
  6.  per-row run_id/window_id align features <-> meta
  7.  all 5 labels present; each run_id maps to exactly one label (print counts)
  8.  t_end > t_start; window_id unique within run; no straddle
  9.  >= 2 distinct run_ids per label
  10. overall PASS, else exit non-zero

Run from repo root: ``python -m feature_extraction.validate_b_outputs``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from simulator.engine_simulator import CONDITIONS
from feature_extraction.pipeline import FEATURE_COLUMNS, META_COLUMNS, OUTPUT_DIR


def validate(features_csv, spectrograms_npy, meta_json):
    checks = []

    def record(ok, msg):
        checks.append((bool(ok), msg))

    features = pd.read_csv(features_csv)
    specs = np.load(spectrograms_npy)
    with open(meta_json) as f:
        meta = json.load(f)

    # 1. meta fs
    record(meta.get("fs") == 25600, f"(1) meta fs == 25600 (got {meta.get('fs')})")

    # 2. row counts
    n_feat = len(features)
    n_npy = specs.shape[0]
    n_meta = len(meta.get("rows", []))
    record(n_feat == n_npy == n_meta,
           f"(2) row counts match: features={n_feat} npy={n_npy} meta={n_meta}")

    # 3. spectrogram ndim / channels
    record(specs.ndim == 4 and specs.shape[1] == 2,
           f"(3) spectrograms ndim==4 & channels==2 (shape={specs.shape})")

    # 4. no NaN/Inf
    feat_vals = features[FEATURE_COLUMNS].to_numpy(dtype=float) \
        if set(FEATURE_COLUMNS).issubset(features.columns) else features.select_dtypes("number").to_numpy()
    feat_clean = np.all(np.isfinite(feat_vals))
    spec_clean = np.all(np.isfinite(specs))
    record(feat_clean and spec_clean,
           f"(4) no NaN/Inf (features_finite={feat_clean}, spec_finite={spec_clean})")

    # 5. feature count + exact column set/order
    expected = META_COLUMNS + FEATURE_COLUMNS
    cols_ok = list(features.columns) == expected
    count_ok = len(FEATURE_COLUMNS) == 74
    record(cols_ok and count_ok,
           f"(5) feature count==74 & exact column order matches "
           f"(count={len(FEATURE_COLUMNS)}, order_ok={cols_ok})")

    # 6. per-row run_id/window_id alignment features <-> meta
    align_ok = True
    for i, mrow in enumerate(meta["rows"]):
        if i >= len(features):
            align_ok = False
            break
        frow = features.iloc[i]
        if str(frow["run_id"]) != str(mrow["run_id"]) \
                or int(frow["window_id"]) != int(mrow["window_id"]):
            align_ok = False
            break
    record(align_ok, "(6) per-row run_id/window_id align features<->meta")

    # 7. all 5 labels present; each run_id -> one label
    labels = set(features["label"].unique())
    all_labels = set(CONDITIONS).issubset(labels)
    per_run_single = features.groupby("run_id")["label"].nunique().max() == 1
    label_counts = features.groupby("label").size().to_dict()
    record(all_labels and per_run_single,
           f"(7) all 5 labels present & one label per run_id; counts={label_counts}")

    # 8. t_end > t_start; window_id unique within run; no straddle
    t_ok = bool((features["t_end"] > features["t_start"]).all())
    uniq_ok = bool((features.groupby("run_id")["window_id"].apply(
        lambda s: s.is_unique)).all())
    record(t_ok and uniq_ok,
           f"(8) t_end>t_start & window_id unique within run "
           f"(t_ok={t_ok}, uniq_ok={uniq_ok})")

    # 9. >= 2 distinct run_ids per label
    runs_per_label = features.groupby("label")["run_id"].nunique()
    runs_ok = bool((runs_per_label >= 2).all())
    record(runs_ok, f"(9) >=2 distinct run_ids per label "
                    f"(min={int(runs_per_label.min())})")

    overall = all(ok for ok, _ in checks)

    print("=" * 60)
    for ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    print("=" * 60)
    print(f"(10) OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


def main():
    ok = validate(
        OUTPUT_DIR / "features.csv",
        OUTPUT_DIR / "spectrograms.npy",
        OUTPUT_DIR / "spectrogram_meta.json",
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
