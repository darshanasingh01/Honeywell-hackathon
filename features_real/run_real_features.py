"""Command-line entry point.

Examples (Windows PowerShell, from inside the features_real folder):
    python run_real_features.py --audio-dir "C:\\data\\audio" --csv "C:\\data\\EngineFaultDB_Final.csv" --out real_outputs
    python run_real_features.py --audio-dir "C:\\data\\audio" --out real_outputs --no-spectrograms
"""
import argparse
import sys

from real_features.config import AudioConfig, CsvConfig
from real_features.pipeline import run_audio, run_csv


def main(argv=None):
    p = argparse.ArgumentParser(description="Real engine audio + CSV feature extraction (outputs are all prefixed real_)")
    p.add_argument("--audio-dir", help="folder containing diesel*/petrol* subfolders of .wav files (searched recursively)")
    p.add_argument("--csv", help="EngineFaultDB_Final.csv")
    p.add_argument("--out", default="real_outputs", help="output folder (default: real_outputs)")
    p.add_argument("--no-spectrograms", action="store_true", help="skip the (large) spectrogram .npz")
    p.add_argument("--win-s", type=float, default=None, help="audio window seconds (default 0.5)")
    p.add_argument("--hop-s", type=float, default=None, help="audio hop seconds (default 0.25)")
    p.add_argument("--highpass", type=float, default=None, help="audio high-pass Hz, 0 = off (default 20)")
    args = p.parse_args(argv)
    if not args.audio_dir and not args.csv:
        p.error("give --audio-dir and/or --csv")

    if args.audio_dir:
        cfg = AudioConfig()
        if args.win_s is not None:
            cfg.win_s = args.win_s
        if args.hop_s is not None:
            cfg.hop_s = args.hop_s
        if args.highpass is not None:
            cfg.highpass_hz = args.highpass
        run_audio(args.audio_dir, args.out, cfg, save_spectrograms=not args.no_spectrograms)
    if args.csv:
        run_csv(args.csv, args.out, CsvConfig())
    return 0


if __name__ == "__main__":
    sys.exit(main())
