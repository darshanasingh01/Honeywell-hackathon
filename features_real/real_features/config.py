"""Settings for the real-data feature pipeline. Change values here (or via CLI flags)."""
from dataclasses import dataclass, field, asdict


@dataclass
class AudioConfig:
    sr: int = 44100                 # recordings are resampled to this if they differ
    win_s: float = 0.5              # window length in seconds
    hop_s: float = 0.25             # hop between windows (50% overlap)
    min_len_s: float = 0.25         # recordings shorter than win_s but >= this get ONE zero-padded window
    highpass_hz: float = 20.0       # removes DC / rumble before everything else (0 disables)
    # log-mel spectrogram
    n_fft: int = 4096               # ~10.8 Hz resolution, useful for low-frequency engine content
    spec_hop: int = 882             # 20 ms
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 16000.0
    n_mfcc: int = 13
    # envelope analysis (band-pass -> Hilbert envelope -> envelope spectrum)
    env_band: tuple = (1000.0, 8000.0)
    env_search_hz: tuple = (5.0, 500.0)
    # relative band energies
    band_edges: tuple = (20.0, 100.0, 300.0, 1000.0, 3000.0, 8000.0, 16000.0, 22050.0)
    # recording-level split
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42


@dataclass
class CsvConfig:
    label_col: str = "Fault"
    win_rows: int = 128             # rows per window
    hop_rows: int = 64
    chunk_rows: int = 2000          # contiguous block unit used for train/val/test assignment
    purge_rows: int = 128           # rows trimmed from both ends of every chunk (leakage gap)
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42


def as_dict(cfg):
    return asdict(cfg)
