# Nine-sensor V2 data handoff

This is a read-only export from the existing validated V2 training handoff. It
does not regenerate a run, retrain a model, or modify the producer/dataset.
The compact files are in `sensor_exports_v2/`:

- `normal_window0.json`
- `misfire_window0.json`
- `overheating_window0.json`
- `index.json`

Each sample is one exact generated `run_id` and `window_id=0`, using the
half-open shared-clock window `[t_start,t_end)`. It contains condition label,
run/window metadata, provenance, 10 sensor-group descriptors, decoded 10 Hz
telemetry samples, indexed 60-2 CKP events, and feature-level fast-sensor
evidence.

The nine primary groups are microphone, knock, CKP, engine RPM, engine load,
MAP, throttle, coolant, fuel trims, and ignition timing. The export also carries
the supporting three-axis accelerometer, for 10 groups in the payload. Units and
nominal rates are repeated in each file. ECU fields are timestamped in seconds:
RPM, normalized load fraction, load percent, MAP kPa, throttle percent,
coolant degC, STFT/LTFT percent, and ignition degrees BTDC. CKP entries contain
timestamp and physical tooth index 0..57; the missing 58/59 positions are the
expected 60-2 geometry gap.

The source handoff retained decoded telemetry and CKP events, but not raw
microphone, knock, or accelerometer arrays. This is explicit in
`waveform_evidence.raw_waveform_available=false` and the index's
`missing_evidence`. Feature summaries remain available for all three fast
waveform groups; they are not raw trace substitutes. No missing trace is
fabricated or zero-filled.

The exporter verifies the complete source handoff before writing samples, then
checks that every telemetry and CKP timestamp lies in the selected window and
that source provenance, run ID, label, and window ID match the feature row.
The files are synthetic-only and are not a diagnostic result or physical sensor
validation.

```powershell
& "$env:TEMP\EngineGuardAI-ml-136-py314\Scripts\python.exe" -B `
  -m ml.synthetic_conditions.export_sensor_v2
```

The command refuses to overwrite an existing export directory. Use a new
`--output` path to create another read-only export.
