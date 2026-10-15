# Synthetic Engine Dataset Schema

## Purpose and limitations

This is a reproducible, illustrative synthetic dataset for a hackathon.
It is not measured engine data and has not been validated against a real
engine. The fault signatures are hypotheses for testing signal-processing
and ML pipelines, not proof of real-world diagnostic accuracy.

## Conditions

`normal`, `misfire`, `knock`, `valve_train_wear`, `overheating`

Each recording has one condition label. Avoid putting windows from the same
recording in both training and test splits.

## Files

For each condition:
- `<condition>_telemetry.csv`
- `<condition>_waveform.csv`

Files are generated in `data/raw/`.

## Telemetry — nominally 10 Hz

| Column | Meaning | Units |
|---|---|---|
| `timestamp_s` | Time since recording start | s |
| `engine_rpm` | Engine speed estimate | rpm |
| `coolant_temp_c` | Coolant temperature | °C |
| `throttle_pct` | Simulated throttle opening | % |
| `engine_load` | Simulated load fraction | 0–1 |
| `stft_pct` | Short-term fuel trim-like value | % |
| `ltft_pct` | Long-term fuel trim-like value | % |
| `ignition_timing_deg` | Simulated spark advance relative to TDC | crank degrees |
| `condition` | Synthetic class label | category |

## Waveform — nominally 4,000 Hz

| Column | Meaning | Units |
|---|---|---|
| `timestamp_s` | Time since recording start | s |
| `microphone_audio` | Synthetic acoustic signal | arbitrary units |
| `knock_sensor` | Synthetic knock/vibration signal | arbitrary units |
| `crank_pulse` | Simplified synthetic crank pulse | binary |
| `accel_x` | Accelerometer X-axis simulation | arbitrary units |
| `accel_y` | Accelerometer Y-axis simulation | arbitrary units |
| `accel_z` | Accelerometer Z-axis simulation | arbitrary units |
| `condition` | Synthetic class label | category |

## Sensor/telemetry categories

The prototype models 11 categories: microphone, knock sensor, crankshaft
position, RPM, coolant temperature, throttle position, engine load, STFT,
LTFT, three-axis accelerometer, and ignition timing. RPM and ignition timing
are treated as ECU-like telemetry, not independent physical sensors.

## Reproducibility

The `seed` argument controls random noise. Identical parameters and seeds
produce identical outputs. Default duration is 10 seconds, telemetry is
10 Hz, and waveform sampling is 4,000 Hz.

## Integration

Person B can extract time/frequency features from waveform windows.
Person C can evaluate classifiers and anomaly detectors.
Person D can visualize telemetry, waveforms, and model outputs.
