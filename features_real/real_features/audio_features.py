"""Signal processing for the real engine audio: windowing, log-mel spectrograms and per-window features."""
import numpy as np
from scipy import signal, stats
from scipy.fft import dct
from scipy.io import wavfile


# ----------------------------------------------------------------------------- I/O
def read_wav(path, target_sr):
    """Read a WAV as float32 mono in [-1, 1], resampled to target_sr if needed."""
    sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max + 1)
    elif x.dtype.kind == "u":
        info = np.iinfo(x.dtype)
        x = (x.astype(np.float64) - (info.max + 1) / 2) / ((info.max + 1) / 2)
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != target_sr:
        g = np.gcd(int(sr), int(target_sr))
        x = signal.resample_poly(x, target_sr // g, sr // g)
    return x.astype(np.float32)


# ----------------------------------------------------------------------------- filtering / windowing
def highpass(x, sr, fc):
    if not fc or fc <= 0:
        return x.astype(np.float64)
    sos = signal.butter(4, fc, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x.astype(np.float64))


def bandpass(x, sr, lo, hi):
    hi = min(hi, 0.49 * sr)
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x.astype(np.float64))


def window_starts(n_samples, win, hop, min_len):
    """Start indices of full windows. A short recording (>= min_len) yields one zero-padded window.
    Returns (starts, padded_flag)."""
    starts = list(range(0, n_samples - win + 1, hop))
    if starts:
        return starts, False
    if n_samples >= min_len:
        return [0], True
    return [], False


def cut(x, start, win):
    seg = x[start:start + win]
    if len(seg) < win:
        seg = np.pad(seg, (0, win - len(seg)))
    return seg


# ----------------------------------------------------------------------------- spectrogram
def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (bins - lo) / (ce - lo)
        down = (hi - bins) / (hi - ce)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def log_mel(seg, sr, fb, n_fft, hop):
    """Log-mel spectrogram (dB) of one window: shape (n_mels, n_frames). No centre padding."""
    seg = seg.astype(np.float64)
    n_frames = 1 + (len(seg) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = seg[idx] * np.hanning(n_fft)[None, :]
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = fb @ power.T
    return 10.0 * np.log10(mel + 1e-10)


# ----------------------------------------------------------------------------- order analysis utility
def order_spectrum(seg, sr, rpm, max_order=20.0, n_bins=400):
    """Magnitude spectrum of a window re-expressed against engine ORDER (multiples of rotation frequency).

    Valid when RPM is roughly constant inside the window. The audio files carry no RPM, so this is not
    called by the pipeline; use it when an RPM value is available (e.g. from the simulator or the CSV).
    Returns (orders, magnitude).
    """
    seg = np.asarray(seg, dtype=np.float64)
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) / len(seg)
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    rot_hz = rpm / 60.0
    orders = np.linspace(0.0, max_order, n_bins)
    return orders, np.interp(orders * rot_hz, freqs, mag)


# ----------------------------------------------------------------------------- per-window features
def band_names(edges):
    return [f"band_{int(lo)}_{int(hi)}" for lo, hi in zip(edges[:-1], edges[1:])]


def feature_names(cfg):
    names = ["rms", "peak", "crest", "zcr", "skew", "kurtosis",
             "spec_centroid", "spec_bandwidth", "spec_rolloff85", "spec_flatness", "spec_entropy", "dominant_hz"]
    names += band_names(cfg.band_edges)
    names += ["env_mod_depth", "env_kurtosis", "env_peak_hz", "env_peak_ratio"]
    names += [f"mfcc{i + 1}_mean" for i in range(cfg.n_mfcc)]
    names += [f"mfcc{i + 1}_std" for i in range(cfg.n_mfcc)]
    return names


def window_features(w_hp, w_bp, sr, logmel_db, cfg):
    """Feature dict for one window. w_hp = high-passed audio, w_bp = band-passed audio (for the envelope),
    logmel_db = that window's log-mel spectrogram."""
    f = {}
    rms = float(np.sqrt(np.mean(w_hp ** 2)))
    peak = float(np.max(np.abs(w_hp)))
    f["rms"] = rms
    f["peak"] = peak
    f["crest"] = peak / (rms + 1e-12)
    f["zcr"] = float(np.mean(np.abs(np.diff(np.signbit(w_hp).astype(np.int8)))))
    f["skew"] = float(stats.skew(w_hp))
    f["kurtosis"] = float(stats.kurtosis(w_hp))

    # whole-window power spectrum (1 Hz resolution for a 1 s window)
    P = np.abs(np.fft.rfft(w_hp * np.hanning(len(w_hp)))) ** 2
    fr = np.fft.rfftfreq(len(w_hp), 1.0 / sr)
    keep = fr >= cfg.band_edges[0]
    P, fr_k = P[keep], fr[keep]
    tot = P.sum() + 1e-20
    cen = float((fr_k * P).sum() / tot)
    f["spec_centroid"] = cen
    f["spec_bandwidth"] = float(np.sqrt(((fr_k - cen) ** 2 * P).sum() / tot))
    f["spec_rolloff85"] = float(fr_k[np.searchsorted(np.cumsum(P), 0.85 * tot)])
    f["spec_flatness"] = float(np.exp(np.mean(np.log(P + 1e-20))) / (P.mean() + 1e-20))
    p = P / tot
    f["spec_entropy"] = float(-(p * np.log2(p + 1e-20)).sum() / np.log2(len(p)))
    f["dominant_hz"] = float(fr_k[np.argmax(P)])
    for name, lo, hi in zip(band_names(cfg.band_edges), cfg.band_edges[:-1], cfg.band_edges[1:]):
        m = (fr_k >= lo) & (fr_k < hi)
        f[name] = float(P[m].sum() / tot)

    # envelope analysis: modulation of the 1-8 kHz band (combustion / knock impulses show up here)
    env = np.abs(signal.hilbert(w_bp))
    mean_env = float(env.mean()) + 1e-12
    f["env_mod_depth"] = float(env.std() / mean_env)
    f["env_kurtosis"] = float(stats.kurtosis(env))
    E = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(len(env))))
    efr = np.fft.rfftfreq(len(env), 1.0 / sr)
    lo, hi = cfg.env_search_hz
    m = (efr >= lo) & (efr <= hi)
    k = int(np.argmax(E[m]))
    f["env_peak_hz"] = float(efr[m][k])
    f["env_peak_ratio"] = float(E[m][k] / (np.median(E[m]) + 1e-12))

    # MFCC (from the log-mel spectrogram, orthonormal DCT-II)
    mf = dct(logmel_db, type=2, axis=0, norm="ortho")[:cfg.n_mfcc]
    for i in range(cfg.n_mfcc):
        f[f"mfcc{i + 1}_mean"] = float(mf[i].mean())
        f[f"mfcc{i + 1}_std"] = float(mf[i].std())
    return f
