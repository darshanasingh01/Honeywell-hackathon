# Real audio audit and handoff

`ml/datasets/real_audio/raw/` is not present in this checkout. The audit tool
therefore reports `source_missing` unless an explicit `--fallback-root` is
provided. An existing, separate HL-CEAD extraction was found at
`ml/datasets/hl_cead/extracted/MLVRG-Car-Engine-Audio`; it can be audited with:

```powershell
& "$env:TEMP\EngineGuardAI-ml-136-py314\Scripts\python.exe" -B `
  -m ml.acoustic.real_audio_audit `
  --fallback-root ml/datasets/hl_cead/extracted/MLVRG-Car-Engine-Audio `
  --output ml/acoustic/real_audio_exports_v2
```

The resulting `manifest.json` records each WAV's relative filename, SHA-256,
duration, sample rate, channels, bit depth, frame count, clipping count and
fraction, corruption status, duplicate group, and `label: "unknown"`. Folder
names and RPM metadata are not promoted to fault labels. The index contains the
aggregate audit and three representative compressed NPZ exports with waveform
and STFT spectrogram arrays.

The compatible acoustic subset follows the existing microphone feature intent:
spectral centroid, bandwidth, 85% rolloff, RMS and zero-crossing summaries.
MFCCs are explicitly marked unavailable when `librosa` is absent. The exporter
never runs the 167-feature classifier, assigns fault labels, modifies WAVs, or
mixes real audio into synthetic V2 training. Real audio exports remain
metadata/features only and are not diagnostic predictions.
