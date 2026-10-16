"""Read-only audit and export for real WAV recordings; never assigns fault labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
from scipy.signal import stft

DEFAULT_ROOT = Path("ml/datasets/real_audio/raw")
KNOWN_ALTERNATE_ROOT = Path("ml/datasets/hl_cead/extracted/MLVRG-Car-Engine-Audio")
DEFAULT_OUTPUT = Path("ml/acoustic/real_audio_exports_v2")
EXPORT_VERSION = "engineguard-real-audio-audit/1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_pcm(raw, width):
    if width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128) / 128.0
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if width == 3:
        values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        integers = values[:, 0].astype(np.int32) | (values[:, 1].astype(np.int32) << 8) | (values[:, 2].astype(np.int32) << 16)
        integers[integers & 0x800000 != 0] -= 1 << 24
        return integers.astype(np.float64) / 8388608.0
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError(f"unsupported PCM sample width: {width} bytes")


def _read_wav(path):
    with wave.open(str(path), "rb") as stream:
        channels, rate, width, frames = stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getnframes()
        raw = stream.readframes(frames)
    if channels <= 0 or rate <= 0 or frames <= 0:
        raise ValueError("empty or invalid WAV metadata")
    samples = _decode_pcm(raw, width)
    if samples.size != frames * channels:
        raise ValueError("truncated PCM payload")
    samples = samples.reshape(frames, channels)
    return samples, {"sample_rate_hz": rate, "channels": channels, "bits_per_sample": width * 8,
                     "frames": frames, "duration_s": frames / rate}


def _compatible_features(audio, rate):
    """Use existing microphone definitions where dependencies permit.

    The runtime environment lacks librosa, so MFCCs are explicitly unavailable;
    this computes the compatible spectral/energy subset with scipy only.
    """
    audio = np.asarray(audio, dtype=float)
    nperseg = min(2048, len(audio))
    frequencies, times, matrix = stft(audio, fs=rate, nperseg=nperseg,
                                       noverlap=nperseg // 2, boundary=None)
    power = np.abs(matrix) ** 2
    total = power.sum(axis=0) + 1e-12
    centroid = (frequencies[:, None] * power).sum(axis=0) / total
    bandwidth = np.sqrt(((frequencies[:, None] - centroid) ** 2 * power).sum(axis=0) / total)
    cumulative = np.cumsum(power, axis=0)
    rolloff = frequencies[np.minimum((cumulative >= cumulative[-1] * 0.85).argmax(axis=0), len(frequencies) - 1)]
    rms = np.sqrt(np.mean(np.square(audio)))
    zcr = np.mean(audio[:-1] * audio[1:] < 0) if len(audio) > 1 else 0.0
    result = {
        "spectral_centroid_mean_hz": float(centroid.mean()), "spectral_centroid_std_hz": float(centroid.std()),
        "spectral_bandwidth_mean_hz": float(bandwidth.mean()), "spectral_bandwidth_std_hz": float(bandwidth.std()),
        "spectral_rolloff_85_mean_hz": float(rolloff.mean()), "spectral_rolloff_85_std_hz": float(rolloff.std()),
        "rms_mean": float(rms), "rms_std": 0.0, "zero_crossing_rate_mean": float(zcr),
        "zero_crossing_rate_std": 0.0,
    }
    return result, {"mfcc_available": False, "mfcc_reason": "librosa is not installed in the audit environment",
                    "compatible_feature_names": sorted(result)}


def audit_wavs(root):
    root = Path(root)
    paths = sorted(p for p in root.rglob("*.wav") if p.is_file()) if root.is_dir() else []
    records, errors, hash_to_paths = [], [], {}
    for path in paths:
        record = {"filename": path.relative_to(root).as_posix(), "label": "unknown"}
        try:
            audio, properties = _read_wav(path)
            features, feature_meta = _compatible_features(audio.mean(axis=1), properties["sample_rate_hz"])
            record.update(properties)
            record["sha256"] = sha256(path)
            # Treat samples within 0.1% of full scale as clipped; PCM max values
            # normalize to slightly below 1.0 (e.g. 32767/32768).
            record["clipping_sample_count"] = int(np.count_nonzero(np.abs(audio) >= 0.999))
            record["clipping_fraction"] = float(np.mean(np.abs(audio) >= 0.999))
            record["corrupt"] = False
            record["compatible_features"] = features
            record["feature_metadata"] = feature_meta
            hash_to_paths.setdefault(record["sha256"], []).append(record["filename"])
        except (OSError, EOFError, ValueError, wave.Error) as exc:
            record.update({"corrupt": True, "error": str(exc)})
            errors.append(record["filename"])
        records.append(record)
    duplicate_groups = [names for names in hash_to_paths.values() if len(names) > 1]
    for record in records:
        record["duplicate_group"] = next((i for i, names in enumerate(duplicate_groups) if record["filename"] in names), None)
    readable = [record for record in records if not record.get("corrupt")]
    return records, {"wav_count": len(paths), "readable_count": len(readable), "corrupt_count": len(errors),
                     "duplicate_group_count": len(duplicate_groups),
                     "duplicate_file_count": sum(len(g) - 1 for g in duplicate_groups),
                     "sample_rate_hz": sorted({r["sample_rate_hz"] for r in readable}),
                     "channel_counts": sorted({r["channels"] for r in readable}),
                     "duration_total_s": float(sum(r["duration_s"] for r in readable)),
                     "duration_min_s": float(min((r["duration_s"] for r in readable), default=0)),
                     "duration_max_s": float(max((r["duration_s"] for r in readable), default=0)),
                     "clipped_file_count": int(sum(r.get("clipping_sample_count", 0) > 0 for r in readable)),
                     "errors": errors}


def _sample_export(path, record, output, sample_index):
    audio, properties = _read_wav(path)
    mono = audio.mean(axis=1)
    # Store normalized waveform and dB spectrogram in a dependency-light NPZ.
    frequencies, times, matrix = stft(mono, fs=properties["sample_rate_hz"], nperseg=min(2048, len(mono)),
                                      noverlap=min(1024, len(mono) // 2), boundary=None)
    db = (20 * np.log10(np.abs(matrix) + 1e-10)).astype(np.float32)
    target = output / f"sample_{sample_index:02d}_{Path(record['filename']).stem}.npz"
    np.savez_compressed(target, waveform=mono.astype(np.float32), spectrogram_db=db,
                        frequencies_hz=frequencies.astype(np.float32), times_s=times.astype(np.float32))
    return {"filename": target.name, "source_filename": record["filename"], "sha256": sha256(target),
            "waveform_samples": int(len(mono)), "spectrogram_shape": list(db.shape)}


def export(root=DEFAULT_ROOT, output=DEFAULT_OUTPUT, fallback_root=None):
    root, output = Path(root), Path(output)
    source_note = None
    if not root.is_dir():
        if fallback_root and Path(fallback_root).is_dir():
            source_note = f"Requested root missing; audited explicit fallback root: {Path(fallback_root).as_posix()}"
            root = Path(fallback_root)
        else:
            return {"status": "source_missing", "requested_root": str(root.as_posix()), "wav_count": 0,
                    "label_policy": "unknown unless verified metadata exists", "exports_created": []}
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    records, summary = audit_wavs(root)
    output.mkdir(parents=True, exist_ok=False)
    readable = [r for r in records if not r.get("corrupt")]
    samples = []
    for sample_index, record in enumerate((readable[0:1] + readable[len(readable)//2:len(readable)//2+1] + readable[-1:]), 1):
        samples.append(_sample_export(root / record["filename"], record, output, sample_index))
    manifest = {"format_version": EXPORT_VERSION, "status": "PASS", "source_root": root.as_posix(),
                "requested_root": str(Path(DEFAULT_ROOT).as_posix()), "source_note": source_note,
                "label_policy": "unknown unless verified metadata exists; no fault labels inferred",
                "real_engine_measurements": True, "summary": summary, "records": records,
                "feature_provenance": {"existing_definition": "ml/acoustic/features.py",
                    "compatible_subset": "scipy spectral centroid/bandwidth/rolloff, RMS and zero-crossing summaries",
                    "mfcc_available": False, "classifier_run": False},
                "sample_exports": samples}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    index = {"format_version": EXPORT_VERSION, "status": "PASS", "source_root": root.as_posix(),
             "label_policy": manifest["label_policy"], "summary": summary, "sample_exports": samples,
             "manifest_sha256": sha256(output / "manifest.json"),
             "limitations": ["Labels are unknown; no fault labels inferred.",
                             "MFCC extraction was unavailable because librosa is not installed.",
                             "Exports are real audio metadata/features only, separate from synthetic V2 training."]}
    (output / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fallback-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.output, args.fallback_root), indent=2))


if __name__ == "__main__":
    main()
