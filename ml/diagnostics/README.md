# Three-sensor diagnostic interface

This package provides a shared JSON envelope and conservative late-fusion rules
for **acoustic/microphone**, **vibration/knock**, and **CKP/crankshaft-position**
inference. It does not train models. RPM is operating context; it is not fault
evidence. The original sensor-level outputs remain visible in every fusion
response.

## Prediction envelope

Every sensor result uses `schema_version: "1.0"` and the contract in
[`prediction.schema.json`](prediction.schema.json). `schema.py` validates the
same fields before fusion. `sensor_type` is restricted to `acoustic`,
`vibration`, and `ckp`; `status` is `available`, `unavailable`, or `error`.
Unsupported RPM, anomaly, and fault predictions stay `null`. Errors carry a
code and message. Unavailable sensors carry a reason in `limitations`.

`prediction_id` identifies an inference request only. It must never be used to
match sensor records. `event_id`, `engine_id`, `session_id`, and interval
timestamps must originate in acquisition metadata. A timestamp interval is
UTC/offset-aware ISO-8601 and must have both endpoints. `signal_quality` may be
`good`, `poor`, `not_assessed`, or `unavailable`; a poor-quality signal is
excluded from fusion evidence.

## Acoustic adapter

From the repository root, the adapter invokes the existing acoustic
`predict.py` using the current Python interpreter and wraps its JSON. It passes
through the predicted RPM, confidence, novelty score, threshold, flag,
limitations, model version, and the full native JSON under `native_output`.
It does not edit or retrain the acoustic pipeline or artifacts. Engine/session
IDs and measurement times are optional flags; the adapter never invents them.
The existing acoustic `fault_prediction` remains null.

```powershell
.\.venv\Scripts\python.exe -m ml.diagnostics.adapters.acoustic path\to\engine.wav
```

Acquisition metadata can be supplied when it is known:

```powershell
.\.venv\Scripts\python.exe -m ml.diagnostics.adapters.acoustic path\to\engine.wav `
  --engine-id engine-17 --session-id session-2026-10-16 --event-id capture-004 `
  --start-utc 2026-10-16T10:00:00Z --end-utc 2026-10-16T10:00:15Z
```

The event and interval values above are examples; only use values recorded by
the acquisition process. The existing `ml/acoustic/predict.py` CLI remains
unchanged and continues to emit its original JSON format.

## Vibration bearing-condition adapter

The NUST adapter can populate the schema's bearing-condition fields, but its
`fault_probabilities` values are uncalibrated model scores, not verified fault
probabilities. The labels describe only the dataset's healthy/faulty bearing
categories; they do not identify mechanism or severity and do not diagnose
combustion knock. The exported timestamps do not verify the paper's sample
rate. Predictions with poor timestamp quality remain visible in
`sensor_results` but provide no eligible anomaly evidence and cannot
corroborate another sensor.

### Acoustic-only example

The numbers below illustrate the envelope shape; they are **not a new measured
inference**. A real adapter response puts the exact source JSON in
`native_output`.

```json
{
  "schema_version": "1.0",
  "prediction_id": "generated-inference-id",
  "event_id": null,
  "sensor_type": "acoustic",
  "status": "available",
  "engine_id": null,
  "session_id": null,
  "measurement_interval": {"start_utc": null, "end_utc": null},
  "rpm": {"value": 1500, "confidence": null, "source": "acoustic_model"},
  "anomaly": {"score": 0.08, "threshold": 0.12, "is_unusual": false, "score_name": "acoustic_novelty_score"},
  "fault_prediction": null,
  "fault_probabilities": null,
  "signal_quality": {"status": "not_assessed", "metrics": {}},
  "model_version": "example-only",
  "limitations": ["Illustrative only; not a measured prediction."],
  "error": null,
  "native_output": {"illustrative": true}
}
```

## Late fusion

Call `fuse_predictions(iterable)` with any subset of sensor envelopes. It
returns `fusion_status` (`single_sensor`, `not_fused`, or `fused`), the original
inputs unchanged under `sensor_results`, and separate `anomaly_evidence` per
sensor. A multi-sensor fusion requires at least two distinct, available,
non-poor-quality sensor results with anomaly outputs and matching,
acquisition-provided `engine_id`, `session_id`, and `event_id`, plus a
non-empty overlap of their measurement intervals. Missing or conflicting
metadata, or non-overlapping intervals, prevents fusion. `prediction_id` is
never part of the match.

Unavailable, error, poor-quality, and quality-unavailable results contribute
no anomaly evidence. `not_assessed` quality is retained as unknown; it is not a quality pass. A
single eligible alert results in `review_recommended`. Alerts from at least
two distinct sensors produce `multi_sensor_corroboration` only after the
metadata/interval gate passes. Otherwise an alert remains a single-sensor
review recommendation. These rules do not establish a fault diagnosis.
Different anomaly scores and thresholds are never averaged, ranked, or
compared. RPM remains visible on each sensor result and is not used as fault
evidence. The fusion-level `fault_prediction` is always null until a compatible
fault-fusion method is validated on appropriately grouped same-engine data.

### Three-sensor example — illustrative only

This example demonstrates data shape and logic, **not measured performance**.
It assumes all three acquisition records genuinely carry matching IDs and
overlapping times. The sensor scores intentionally use different scales and are
kept separate. Vibration and CKP are currently untrained/unavailable in this
project, so this is not an actual project output.

```json
{
  "illustrative": true,
  "fusion_status": "fused",
  "event_id": "capture-004",
  "fusion_decision": "multi_sensor_corroboration",
  "fault_prediction": null,
  "anomaly_evidence": [
    {"sensor_type": "acoustic", "score_name": "acoustic_novelty_score", "score": 0.08, "threshold": 0.07, "is_unusual": true},
    {"sensor_type": "vibration", "score_name": "vibration_anomaly_score", "score": 12.4, "threshold": 10.0, "is_unusual": true},
    {"sensor_type": "ckp", "score_name": "ckp_anomaly_score", "score": 0.002, "threshold": 0.001, "is_unusual": true}
  ],
  "sensor_results": ["Full validated sensor envelopes are retained here in a real response."],
  "limitations": ["Illustrative only; anomaly agreement is not a confirmed diagnosis."]
}
```

## Adding sensors later

The current registry is intentionally limited to the three project sensors.
Each future sensor should receive its own schema-approved adapter and
sensor-specific result namespace. Do not combine recordings from separate
engines or sessions to fill missing modalities. Expansion to nine sensors
should add explicit sensor types and validation only after the project approves
those channels and their evidence contracts.
