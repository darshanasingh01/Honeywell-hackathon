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

### Six ECU-style signals for integration

The following six fields form the initial ECU-style telemetry interface for the simulator. They are synthetic values, not decoded CAN frames or measurements from a production engine.

| Signal | CSV field | Unit / scale |
|---|---|---|
| Engine speed | `engine_rpm` | rpm |
| Coolant temperature | `coolant_temp_c` | °C |
| Throttle position | `throttle_pct` | % |
| Engine load | `engine_load` | Normalized scale, 0–1 |
| Short-term fuel trim | `stft_pct` | % |
| Long-term fuel trim | `ltft_pct` | % |

**Integration notes**
- Preserve these exact column names when consuming the generated telemetry.
- `timestamp_s` provides the telemetry time base; the default sampling rate is 10 Hz.
- `condition` is the synthetic ground-truth label for the generated recording.
- `ignition_timing_deg` is an additional ECU-style telemetry field outside the six-signal interface.
- These fields are simulated directly by the generator. Real CAN integration would require a defined message/frame layout, signal scaling, offsets, and decoding rules.
- Confirm the schema and units with Person B before feature extraction or windowing.

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
