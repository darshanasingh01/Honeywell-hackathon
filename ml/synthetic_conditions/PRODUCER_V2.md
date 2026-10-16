# Isolated synthetic V2 producer

`producer_v2.py` adds a complete ML-owned producer without editing or exporting
teammate source. Three modules are loaded from approved immutable Git blobs at
`0ade8c2326d76bf104ba42613f1ea771558c6af8`: engine simulator, signal DSP, and
PID codec. The source commit must already exist locally. No network fetch,
branch changes, source patching, model fitting or dashboard work occurs.

## Contract reconciliation

`schema_v2.json` is authoritative, version `engineguard.ml-features/167-v2.1`.
The historical draft `schema_9sensor_v2.json` has the same 167 ordered names.
Neither schema is modified by this integration.

| Area | Historical draft | Implemented contract / producer |
|---|---|---|
| Version | synthetic-conditions-features/167-v2 | engineguard.ml-features/167-v2.1 |
| Metadata | 22 fields | 24; adds processor revision and timestamp resolution |
| Numeric types | implicit | explicit float64 and integer counts |
| CV units | some CKP CVs incorrectly seconds/RPM | dimensionless |
| Standard deviations | some signed inherited bounds | nonnegative |
| CKP quality | minimum four samples/tooth | ten timestamp ticks/tooth by default |
| Legacy crank/gated quality | availability only | good CKP required |
| Unobservable quality diagnostics | ambiguous | may be null on poor-quality input |
| Window convention | not explicit | shared clock, half-open intervals |
| Provenance | V1 source reference | actual processor identity/revision required too |
| Missingness | null if unavailable/invalid | enforced by validator; nominal dataset requires all finite |

The contract's `status` text and README_V2's training-blocker paragraph describe
Phase 1 and are now historical: this phase provides a synthetic producer and
separately validated local dataset. The schema status string is not an
eligibility or validation flag. The manifest's legacy `knock_cycle_*` note
refers to the standalone adapters; this producer supplies the required gating.
No existing schema or Phase 1 file was rewritten to change those notes.

All 167 paths are enumerated individually in each generated manifest:

| Family | Count | Derivation |
|---|---:|---|
| microphone | 19 | unchanged approved simulator waveforms and DSP |
| knock, including knock-cycle features | 22 | unchanged approved simulator waveform/DSP plus approved 5–10 kHz cycle gate |
| three acceleration axes | 57 | unchanged approved simulator waveforms and DSP |
| legacy ECU telemetry | 35 | approved seven-PID byte roundtrip and statistics |
| legacy crank | 3 | full revolution RPM mean/std and period CV from timestamped indexed CKP |
| MAP | 5 | new declared pressure model, PID 0x0B byte quantization, ecu_v2 window statistics |
| load percentage | 5 | PID 0x04 percent, same decoded samples as legacy load fraction |
| CKP timing / quality | 21 | ckp_v2 timestamped 60-2 processor |

No feature has a placeholder fallback. Missing/poor acquisition fails nominal
generation rather than becoming a zero-valued physical measurement. Tests
exercise dropped pulses and inadequate clock resolution independently of the
five condition labels.

The saved V2 Random Forest retains all 167 predictors, including two features
that are constant across the training handoff (`ckp_valid_tooth_fraction` and
`ckp_unexpected_gap_count`). Four predictors have zero split importance in the
saved fit (`accel_x_mid_1k_5k_energy`, `tel_ltft_pct_min`,
`ckp_valid_tooth_fraction`, and `ckp_unexpected_gap_count`). They remain in the
contract for schema stability and must not be removed without a new version.

## Operating assumptions and synchronization

The approved simulator supplies five seeded condition trajectories and all
microphone, knock, and three-axis accelerometer waveforms at 25.6 kHz. CKP is
event-driven timestamped 60-2 tooth data quantized to a 100 kHz clock. RPM is
linearly interpolated from its 10 Hz ECU
trajectory. The cumulative crank phase uses the approved simulator's convention
(one phase increment at sample zero), so tooth-zero markers align with its
original crank/firing phases within waveform/timestamp quantization. Pitch
crossings are interpolated in time and rounded to a 100 kHz clock; teeth 58/59
are absent. Expected 57-to-0 gaps span three pitches. Unexpected gaps are
acquisition failures, not simulated mechanical diagnoses.

Legacy V2 crank statistics use all complete same-tooth revolution intervals,
matching ckp_v2's cycle definition, not every tooth as a revolution. Gated knock
uses one tooth-zero marker per revolution, preserving the approved V1 gate
design. It does not claim to gate both combustion events or infer cam phase.
The simulator's slow misfire RPM modulation is retained; no new torque-resolved
misfire physics is invented.

MAP assumes a naturally aspirated spark-ignition intake at 101.325 kPa ambient:

`p = ambient * (0.18 + 0.65*load + 0.17*throttle_pct/100)
     * (1 - 0.08*(RPM-1800)/6200)`

Clip to [15, ambient] kPa, then quantize to the PID integer kPa byte. Higher
load/opening reduces vacuum; increased speed at fixed opening increases pumping
demand. This is an interpretable illustrative operating-point surrogate,
not a calibrated manifold, turbocharger, diesel, altitude or transient model.
It uses the common source trajectory, never a direct condition-to-MAP label map.

All ECU payloads are sampled synchronously at 10 Hz without synthetic transport
latency. Approved encode/decode math is reused; no new physical CAN connection
or full frame parser is claimed. This intentionally differs from V1's staggered
request/reply timestamps and unbounded nearest merge. New windows keep the
established 0.5 s length / 0.25 s hop, with exact exclusive end time rather than
the V1 last-sample inclusive endpoint. No interpolation of decoded ECU
measurements is performed; the 0.2 s staleness policy is enforced.

The 4 kHz project target does not match this verified V2 contract. Downsampling
the existing 25.6 kHz waveforms to 4 kHz would invalidate the retained
1â€“5 kHz spectral feature definitions because the Nyquist frequency would be
2 kHz, and the 1â€“5 kHz bands could no longer be measured. Any future resampled
contract requires new feature definitions, schema versioning, and a separately
trained model.

## Reproduce and verify

Use the supported Python environment (the repository .venv is currently broken):

```powershell
& "$env:TEMP\EngineGuardAI-ml-136-py314\Scripts\python.exe" -B `
  -m ml.synthetic_conditions.producer_v2
```

The command refuses existing output roots. To reproduce, supply `--output` with
a new directory; it will not overwrite previous data. It first generates and
validates a five-run, one-second smoke set using seed 3100, including a CSV
roundtrip. Only then does it generate six ten-second runs per condition using
seeds 3200..3205. Namespaces distinguish smoke and training groups. Seeds differ
from retained V1/OOD datasets; this is not physical engine independence.

Each dataset directory contains features.csv, manifest.json, validation.json,
and run_evidence.json. Evidence retains decoded telemetry, actual PID payloads,
timestamped indexed CKP edges, per-window quality checks, seeds and condition.
Large raw waveforms are reproducible from the pinned source/seed and are not
duplicated. Source blob hashes, ML module/schema hashes, software versions,
configuration, every feature path, and feature CSV hash are recorded.

```powershell
& "$env:TEMP\EngineGuardAI-ml-136-py314\Scripts\python.exe" -B `
  -m ml.synthetic_conditions.verify_producer_v2 `
  ml/datasets/synthetic_conditions_v2/training
```

The verifier checks serialized schema/order, provenance and source hashes,
evidence alignment, clock quantization and reconstructed CKP/MAP/load features.
It also checks prospective five-fold run-group splits for disjoint runs and all
classes when there are enough runs; it never fits a classifier. Data is already
covered by repository ignore rules and requires separate artifact delivery.

All outputs are synthetic-only. No per-class classifier metrics, calibration,
real-engine accuracy, robustness to physical acquisition failures, or model
promotion is claimed. V2 classifier training remains outside this phase.
