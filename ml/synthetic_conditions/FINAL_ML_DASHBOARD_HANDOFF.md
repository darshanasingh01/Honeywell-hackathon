# EngineGuard AI — Final ML Dashboard Handoff

This is a read-only handoff for dashboard integration. It references existing
artifacts and does not regenerate data or alter models.

## V2 synthetic 167-feature inference

Use `ml.synthetic_conditions.predict_v2` with the ignored artifact:

```
ml/saved_models/synthetic_conditions_167_v2_baseline.joblib
```

The artifact SHA-256 is:

```
01f693c753d3439f0eb4b7660aac4d95ab339529d4f5a53bc4435c2e11966db3
```

Input is one JSON object containing **exactly** the 167 feature names, in the
order defined by `ml/synthetic_conditions/schema_v2.py` and
`schema_v2.json`. Values must be finite numeric values. Missing or extra names,
booleans, strings, NaN, and infinity are rejected.

Example command (where `normal.json` contains one object from
`v2_examples.json`):

```powershell
python -m ml.synthetic_conditions.predict_v2 normal.json
```

For a single example, pass one of the objects in `v2_examples.json` as the
input file. The JSON response contains `predicted_condition`, one score for
each class, model metadata, dataset provenance, and limitations. Scores are
uncalibrated RF vote fractions, not fault probabilities.

The five valid class labels are `normal`, `knock`, `misfire`, `overheating`,
and `valve_train_wear`. The checked examples predict:

| Example | Expected prediction |
| --- | --- |
| `normal` | `normal` |
| `misfire` | `misfire` |
| `overheating` | `overheating` |

Model loading rejects a missing artifact and rejects bundles whose schema,
feature list, labels, or model version do not match the V2 contract. Input
contract failures are returned as `ValueError` by the Python API and as a CLI
argument error by `predict_v2.py`.

This diagnosis is based on simulated engine-condition data only. It does not
establish real-engine accuracy or calibrated confidence.

## Synthetic nine-sensor export

The dashboard-ready, read-only index is:

```
ml/synthetic_conditions/sensor_exports_v2/index.json
```

Its three synchronized samples are:

```
ml/synthetic_conditions/sensor_exports_v2/normal_window0.json
ml/synthetic_conditions/sensor_exports_v2/misfire_window0.json
ml/synthetic_conditions/sensor_exports_v2/overheating_window0.json
```

The export covers the nine primary groups (microphone, knock, CKP, RPM, engine
load, MAP, throttle, coolant, fuel trims, and ignition timing) plus supporting
three-axis accelerometer data. It includes timestamped ECU telemetry and CKP
events, feature summaries for fast sensors, units, rates, run/window IDs, and
provenance. Raw microphone, knock, and accelerometer waveforms are explicitly
unavailable in this source handoff and were not fabricated.

These labels and measurements are synthetic simulator outputs and must remain
separate from real-audio evidence.

The verified synthetic acquisition rates are 25.6 kHz for microphone, knock,
and supporting accelerometer waveforms; event-driven CKP timestamps with
100 kHz clock quantization; and 10 Hz for all ECU telemetry. The model keeps
all 167 predictors, including the constant CKP features
`ckp_valid_tooth_fraction` and `ckp_unexpected_gap_count`; four features have
zero split importance in the saved Random Forest, but none may be removed from
the contract. A 4 kHz resample would make the retained 1–5 kHz features
invalid because its Nyquist frequency is only 2 kHz.

## Real audio export

The real-audio dashboard index is:

```
ml/acoustic/real_audio_exports_v2/index.json
```

It references:

```
ml/acoustic/real_audio_exports_v2/manifest.json
ml/acoustic/real_audio_exports_v2/sample_01_0001.npz
ml/acoustic/real_audio_exports_v2/sample_02_0003.npz
ml/acoustic/real_audio_exports_v2/sample_03_0001.npz
```

The manifest covers 152 readable WAV recordings. Their labels are all
`unknown`; no fault labels were inferred. The NPZ files contain waveform and
spectrogram evidence plus compatible acoustic features. This is real,
unlabeled audio metadata/features, not a diagnosis, and must not be passed to
the 167-feature classifier as a complete input row.

## Real OBD-II telemetry export

The separate dashboard-ready telemetry file is:

```
ml/datasets/real_obd_carobd_v1/dashboard_timeseries.json
```

It contains two real recorded trajectories, `idle1` and `drive1`, with 120
records each. Each record preserves the source timestamp as `timestamp_s`
(the upstream column is misspelled `ENGINE_RUN_TINE`) and retains repeated
timestamps; no resampling, interpolation, or imputation was applied.

The real-audio export is 44.1 kHz source audio, while the real OBD-II export is
nominally 1 Hz. Both remain separate from synthetic diagnosis and are not
inputs to the saved 167-feature model.

The eight mapped ECU fields are:

| Field | Export unit |
| --- | --- |
| `engine_rpm` | rpm |
| `engine_load_percent` | % |
| `map_kpa` | kPa |
| `throttle_percent` | % |
| `coolant_temperature_degC` | degC |
| `short_term_fuel_trim_percent` | % |
| `long_term_fuel_trim_percent` | % |
| `ignition_timing_advance_degBTDC` | degBTDC |

The source CSV headers do not declare units. The export records these mapped
units as standard OBD interpretations and marks the source units as unknown.
Provenance, source CSV SHA-256 values, source repository, vehicle, acquisition
method, source columns, sampling notes, and missingness policy are in:

```
ml/datasets/real_obd_carobd_v1/manifest.json
```

The dashboard should render these trajectories as a separate **real ECU
telemetry** view. Keep them distinct from synthetic five-class diagnosis and
from real unlabeled audio. Do not attach a synthetic condition label, send
these records to the 167-feature classifier, or combine them with audio-only
features. The telemetry has no verified fault labels and is not a diagnostic
evaluation set.

The V2 model artifact is Git-ignored by `*.joblib` and must be transferred
separately when enabling synthetic inference:

```
ml/saved_models/synthetic_conditions_167_v2_baseline.joblib
```

Use its SHA-256 documented above to verify the delivered copy. The real OBD
raw CSVs and JSON export are separate data artifacts and should be delivered
alongside their manifest when the dashboard needs the telemetry view.

Run `python -B -m unittest tests.test_final_ml_dashboard_handoff -v` to verify
all referenced files, hashes, model loading, and example predictions.
