"""Inspect WAV metadata and folder-derived class counts without loading audio arrays."""

from __future__ import annotations

import argparse
import re
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List WAV sample rates, durations, channels, and folder-derived classes."
    )
    parser.add_argument("root", type=Path, help="Dataset directory to scan recursively")
    parser.add_argument(
        "--class-depth",
        type=int,
        default=1,
        help=(
            "Directory depth relative to root used as the class label: 1 means the first "
            "folder below root; 0 groups everything as 'unlabelled' (default: 1)."
        ),
    )
    parser.add_argument(
        "--label-regex",
        help=(
            "Optional regex applied to the relative WAV path. The first capture group is "
            "used when present, otherwise the full match. Overrides --class-depth."
        ),
    )
    return parser.parse_args()


def class_for(path: Path, root: Path, depth: int, label_regex: re.Pattern[str] | None) -> str:
    relative = path.relative_to(root)
    if label_regex is not None:
        match = label_regex.search(relative.as_posix())
        if match:
            return match.group(1) if match.lastindex else match.group(0)
        return "unmatched"
    if depth == 0:
        return "unlabelled"
    folders = relative.parts[:-1]
    if len(folders) >= depth:
        return "/".join(folders[:depth])
    return "unlabelled"


def inspect(root: Path, depth: int, label_regex: re.Pattern[str] | None) -> int:
    root = root.expanduser().resolve()
    if not root.is_dir():
        print(f"Error: dataset root is not a directory: {root}", file=sys.stderr)
        return 2

    wav_paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")
    if not wav_paths:
        print(f"No WAV files found under: {root}")
        return 1

    counts: Counter[str] = Counter()
    rates: Counter[int] = Counter()
    durations: defaultdict[str, float] = defaultdict(float)
    bad_files = 0

    print("relative_path\tclass\tsample_rate_hz\tchannels\tbits_per_sample\tduration_s")
    for path in wav_paths:
        label = class_for(path, root, depth, label_regex)
        try:
            with wave.open(str(path), "rb") as wav:
                rate = wav.getframerate()
                channels = wav.getnchannels()
                width = wav.getsampwidth() * 8
                frames = wav.getnframes()
                duration = frames / rate if rate else 0.0
        except (OSError, EOFError, wave.Error) as exc:
            print(f"Warning: cannot read {path.relative_to(root)}: {exc}", file=sys.stderr)
            bad_files += 1
            continue

        print(
            f"{path.relative_to(root).as_posix()}\t{label}\t{rate}\t{channels}\t{width}\t{duration:.3f}"
        )
        counts[label] += 1
        rates[rate] += 1
        durations[label] += duration

    print("\nClass distribution (readable WAV files):")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]} files, {durations[label]:.3f} s total")
    print("Sampling-rate distribution:")
    for rate, count in sorted(rates.items()):
        print(f"  {rate} Hz: {count} files")
    print(f"Total WAV files: {len(wav_paths)}; readable: {sum(counts.values())}; unreadable: {bad_files}")
    return 0 if counts else 1


def main() -> int:
    args = parse_args()
    if args.class_depth < 0:
        print("Error: --class-depth must be zero or greater", file=sys.stderr)
        return 2
    try:
        label_regex = re.compile(args.label_regex) if args.label_regex else None
    except re.error as exc:
        print(f"Error: invalid --label-regex: {exc}", file=sys.stderr)
        return 2
    return inspect(args.root, args.class_depth, label_regex)


if __name__ == "__main__":
    raise SystemExit(main())
