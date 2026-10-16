# Three-sensor diagnostic interface

The diagnostics package connects the acoustic/microphone, vibration/knock, and
CKP/crankshaft-position model outputs. It preserves the sensor results and
reports evidence by modality. It does not combine RPM estimates, bearing
condition scores, or unlike anomaly scales into a single probability.

## Prediction envelope and provenance

Each result follows [`prediction.schema.json`](prediction.schema.json) and is
checked by `schema.validate_prediction`. The schema limits `sensor_type` to
`acoustic`, `vibration`, or `ckp`. Status is `available`, `unavailable`, or
`error`; unsupported fields stay null. `native_output` retains each adapter's
source JSON, including scores, confidence, class outputs, quality details, and
model limitations.

The optional `provenance` object has `kind` (`acquisition`, `dataset`,
`synthetic`, or `unknown`) and `source_id`. Adapters default to unknown. For
event-level fusion, the acquisition process must supply `kind: "acquisition"`
and a matching, non-empty `source_id` for each sensor stream. Training dataset
names or synthetic provenance do not establish synchronized acquisition.
The same acquisition record must also supply matching `engine_id`,
`session_id`, `event_id`, and timezone-aware measurement intervals. Prediction
IDs identify inference requests only and never establish synchronization.

The adapter functions accept provenance only when the caller has trustworthy
source metadata. Their command-line interfaces accept `--provenance-kind` and
`--provenance-source-id`; supplying these flags asserts the provenance stated
by the acquisition system. The CKP adapter always keeps an input explicitly
marked synthetic as synthetic, regardless of caller arguments.

## Unified interface

Use `fuse_sensor_outputs` with any subset of named sensor envelopes:

```python
from ml.diagnostics.fusion import fuse_sensor_outputs

diagnostic = fuse_sensor_outputs(
    acoustic=acoustic_envelope,
    vibration=vibration_envelope,
    ckp=ckp_envelope,
)
```

Inputs are schema-validated, and a sensor passed in the wrong named slot is
rejected. Missing sensors appear in `missing_sensors`; unavailable and error
outputs remain in `sensor_results` but provide no evidence. Repeated sensor
types remain visible in the generic `fuse_predictions` interface but are not
counted as independent evidence.

Anomaly scores are shown as separate per-sensor entries. Poor or unavailable
quality excludes a result from anomaly evidence. `not_assessed` quality can be
shown for single-sensor review but does not qualify for actual multi-sensor
fusion. `fusion_status` is:

- `single_sensor` when one result was supplied.
- `fused` when at least two distinct good-quality anomaly outputs exist and
  every available good-quality supplied signal has matching acquisition
  provenance, engine/session/event IDs, and a common overlapping interval.
- `not_fused` when those checks fail or evidence is insufficient.

One available alert yields `review_recommended`. Alerts from two or more
distinct eligible modalities yield `multi_sensor_corroboration` only after the
event-level gate passes. This means the sensors independently flag unusual
signals; it is not a confirmed fault diagnosis. The fusion-level
`fault_prediction` remains null. RPM is operating context only.

## Sensor-specific interpretation

- **Acoustic:** the existing model supplies RPM classification and an
  RPM-conditioned novelty score. Quality is `not_assessed` by the current
  inference CLI, so it remains visible but cannot qualify for multi-sensor
  fusion until acquisition quality is assessed.
- **Vibration:** the current classifier predicts NUST healthy/faulty
  bearing-condition labels. Its class scores are uncalibrated and do not
  diagnose combustion knock. Poor signal quality prevents fusion.
- **CKP:** the current estimator supplies RPM only. It does not currently
  produce an anomaly or fault output. Its generated test waveforms are
  synthetic and cannot be labeled as real sensor captures.

These models were developed from separate datasets; their existing evaluations
do not demonstrate synchronized measurements, same-engine fusion, or
real-world multi-sensor fault-detection accuracy.

## Examples

The dashboard example is in
[`examples/three_sensor_demo.json`](examples/three_sensor_demo.json). It is
explicitly illustrative and marked `not_fused`: its three values represent
separate source origins and have no shared acquisition provenance. It shows
how individual outputs remain visible when fusion is withheld. It is not a
measured model response.

An acoustic-only adapter invocation remains supported:

```powershell
.\.venv\Scripts\python.exe -m ml.diagnostics.adapters.acoustic path\to\engine.wav
```

Without explicit acquisition metadata the adapter uses unknown provenance and
null engine/session/event/time fields; the wrapper does not fabricate them.
The original `ml/acoustic/predict.py` CLI and acoustic model artifacts are
unchanged.

## Tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_diagnostics.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
