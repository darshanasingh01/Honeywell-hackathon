"""Generate the raw dataset: 6 seeds x 5 conditions.

For each (condition, seed) run this writes:
  data/raw/<cond>_telemetry_s<seed>.csv
  data/raw/<cond>_waveform_s<seed>.csv
  data/raw/<cond>_can_s<seed>.log      (candump-format CAN frames)

Run from repo root: ``python -m feature_extraction.generate_dataset``
"""
from __future__ import annotations

from pathlib import Path

from simulator.engine_simulator import CONDITIONS, simulate_engine
from feature_extraction.can_codec import telemetry_to_can, write_candump_log

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

SEEDS = (0, 1, 2, 3, 4, 5)
DURATION_S = 10.0
SIGNAL_HZ = 25600
TELEMETRY_HZ = 10


def generate(out_dir=RAW_DIR, seeds=SEEDS, conditions=CONDITIONS,
             duration_s=DURATION_S):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_runs = 0
    for cond in conditions:
        for seed in seeds:
            telemetry, waveform = simulate_engine(
                cond, duration_s, TELEMETRY_HZ, SIGNAL_HZ, seed
            )
            telemetry.to_csv(out_dir / f"{cond}_telemetry_s{seed}.csv", index=False)
            waveform.to_csv(out_dir / f"{cond}_waveform_s{seed}.csv", index=False)

            can_log = telemetry_to_can(telemetry)
            write_candump_log(can_log, out_dir / f"{cond}_can_s{seed}.log")

            n_runs += 1
            print(f"{cond:18s} seed={seed}  "
                  f"telemetry={len(telemetry):4d}  waveform={len(waveform):7d}  "
                  f"can_frames={len(can_log):5d}")
    print(f"\nGenerated {n_runs} runs into {out_dir}")
    return n_runs


def main():
    generate()


if __name__ == "__main__":
    main()
