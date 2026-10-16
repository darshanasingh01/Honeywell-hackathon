# Synthetic CKP analysis dataset

This package generates a reproducible **synthetic** CKP waveform set by calling
the teammate's actual `generate_ckp` function from
`simulator/ckp_model.py`. The implementation is loaded read-only from pinned
sim branch commit `92e69cf6b87c1417c87a61998d391598e53d7081`; each metadata file
records that commit and the SHA-256 of the source. The script neither copies
nor edits simulator files. The pinned commit must be present in the local Git
object database; fetch `origin sim` if it is absent.

## Generate

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe ml\ckp\generate_synthetic_dataset.py
```

Outputs go to `ml/datasets/ckp/synthetic/`, which is Git-ignored. Each CSV has
`time_s`, `crank_angle_deg`, `rpm_ground_truth`, and `ckp_pulse` columns. Each
CSV has an adjacent `.metadata.json` sidecar; `manifest.json` summarizes the
cases. Metadata records synthetic provenance, RPM truth/profile, sample rate,
60-2 wheel geometry, seed, pinned generator provenance, and pulse-validation
results. The current CKP generator is deterministic and adds no random noise;
the seed is recorded for run traceability and does not alter its output.

Cases include constant 800, 1500, and 3000 RPM at 100 kHz, a variable-RPM
profile at 100 kHz, 4000 Hz stress cases at each constant RPM, and a deliberately
under-resolved 3000 RPM / 1000 Hz diagnostic case. The lower-rate diagnostic is
expected to miss pulse events and is explicitly marked unusable.

## Pulse validation and limits

The validator compares observed pulses with the ideal number of valid tooth
slots crossed on the generated angle sequence, counts missing-tooth gaps when
angular resolution can distinguish them, and reports maximum angle step and
samples per tooth. High-rate cases are reference cases only for this synthetic
generator. At 3000 RPM and 4000 Hz, the model has about 1.33 samples per nominal
tooth; timing is coarse even if pulse count is preserved. The 1000 Hz case is
expected to be unusable because sample steps can skip multiple tooth slots.

The model returns a sampled binary signal, not a voltage waveform. It is an
angle-based simplified 60-2 pulse model, not a calibrated production CKP sensor
or verified physical acquisition. These records must not be called real sensor
measurements and cannot establish real-world RPM accuracy, fault detection, or
vehicle generalization. No fault labels or classifier are created here.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_ckp_synthetic_dataset.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_ckp.py"
```

## RPM analysis method

Run the offline reference comparison and save its report:

```powershell
.\.venv\Scripts\python.exe ml\ckp\analyze.py --evaluate-dataset `
  --output ml\evaluation\ckp_rpm_synthetic.json
```

The estimator finds rising pulse edges, classifies the normal missing-tooth
interval as three nominal tooth pitches (18 degrees for 60-2), and estimates
RPM over ten tooth intervals to reduce sample-quantization error. Evaluation
compares each window estimate with synthetic RPM truth interpolated at the
window midpoint and reports MAE/RMSE in RPM and absolute percentage error. The
diagnostics adapter deliberately omits ground-truth comparison fields. It also
leaves engine/session IDs and UTC acquisition times null because synthetic
files do not contain real acquisition provenance.

The estimator returns unavailable for fewer than two observed samples per
tooth, invalid timestamps, unexplained pulse intervals, or a missing-gap
sequence that does not match 60-2 geometry. It marks fewer than ten samples
per tooth poor quality. These are software safeguards, not a guarantee that
all physical sensor failures can be identified.

Sampled edge times are quantized to the sampling grid, and RPM is estimated
over a finite ten-tooth window. The 3000 RPM reference therefore has a small
nonzero error even with complete pulse detection; tests allow ±5 RPM to cover
that discretization without rounding or clamping the estimate. A raw detected
edge spacing alone cannot always identify undersampling: adjacent high samples
may merge, and missing pulses can make observed intervals appear longer. The
4000 Hz / 3000 RPM stress waveform is therefore judged by conservative pulse
pattern and missing-gap checks, with unavailable status required even if the
most specific safety reason varies. No synthetic RPM ground truth is used in
the estimator's validation path.

Generator validation recorded 774/774, 1450/1450, 2900/2900, and 1987/1987
pulses for the 800, 1500, 3000 RPM and variable-RPM reference cases. The 4000
Hz cases preserve pulse counts but have limited timing resolution; the 3000
RPM / 1000 Hz case missed 1900 of 2900 expected pulses and is unusable. RPM
estimation errors are reported by the offline comparison command above; these
are software-validation results on synthetic generator data only.
