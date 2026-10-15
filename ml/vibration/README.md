# Engine vibration analysis

This package retains the sample-rate-aware signal tools and unlabeled novelty
framework in `signals.py`, `features.py`, and `train_novelty.py`. It also has an
exploratory, recording-level NUST journal-bearing condition classifier. The
classifier uses only Channel 3 sample-value statistics; it does not use the
unverified CSV rate, spectral features, fixed-duration windows, or synthetic
labels.

## NUST data status and limitations

The local small archive is expected at
`ml/datasets/vibration/nust/extracted/`. Its CSV directory labels are
`healthy_bearing` and `faulty_bearing`; the fault mechanism and severity are
not specified. The accompanying paper identifies Channel 3 as the bearing
housing accelerometer signal and reports units of `g`, but the CSV does not map
the numbered channels to X/Y/Z.

The paper reports 3,200 Hz, but the exported CSV timestamps have a median
positive increment near 0.0034 seconds, repeated initial rows, and larger
gaps. Thus the CSVs do not verify a uniform 3,200 Hz signal. Time-domain
features summarize exported sample values and are not duration-weighted.
Evaluation is recording-held-out on one engine. All healthy records are dated
May 27, so both evaluation scenarios below can be date-confounded. Neither
scenario measures unseen-session or unseen-engine generalization. The model is
a bearing-condition classifier, not a combustion-knock classifier.

## Train and evaluate

From the repository root in VS Code PowerShell:

```powershell
.\.venv\Scripts\python.exe ml\vibration\train_bearing_condition.py
```

The command deduplicates exact file contents before splitting, removes an
identical repeated first sample, extracts one Channel 3 feature vector per CSV,
and runs five-fold stratified recording-level evaluation:

- **A:** all unique recordings (expected 43 healthy, 90 faulty).
- **B:** May 27 recordings only (expected 43 healthy, 33 faulty).

Each fold fits `StandardScaler` and logistic regression on its training
recordings only. The JSON report contains OOF and per-fold confusion matrices,
balanced accuracy, macro-F1, faulty-class precision/recall, fold variability,
and majority-class baselines. Only after both evaluations complete does the
script fit the final model on all unique recordings.

Default outputs are `ml/saved_models/vibration_nust_bearing_condition.joblib`
and `ml/evaluation/vibration_nust_bearing_condition.json`. Both paths follow
existing ignore rules for generated artifacts/reports.

To compare logistic regression with the fixed class-balanced random-forest
candidate on the exact same five-fold recording splits, run:

```powershell
.\.venv\Scripts\python.exe ml\vibration\compare_bearing_models.py
```

This evaluation-only command runs both all-recording and May 27 subsets, fits
the scaler inside each training fold, and writes only
`ml/evaluation/vibration_nust_model_comparison.json`. It does not tune on
held-out folds or save/replace a model. Because the dataset has no engine or
session IDs, folds are recording-level rather than engine/session-grouped.

## Recording-level inference

```powershell
.\.venv\Scripts\python.exe ml\vibration\predict_bearing_condition.py `
  "ml\datasets\vibration\nust\extracted\Engine Journal Bearings Dataset 2\Engine Journal Bearings Dataset\Faulty Bearings Dataset\1000 RPM\0% Humidity\-10 deg Celsius\1st at -10 2022Jun04-2239-0005.csv" `
  --model ml\saved_models\vibration_nust_bearing_condition.joblib
```

The native JSON includes `predicted_bearing_condition`, uncalibrated class
probabilities, model version, input channel, timestamp-quality metrics, and
limitations. It does not invent acquisition IDs or timestamps.

## Diagnostics adapter

To return the shared schema envelope:

```powershell
.\.venv\Scripts\python.exe -m ml.diagnostics.adapters.vibration `
  "path\to\nust_recording.csv" `
  --model ml\saved_models\vibration_nust_bearing_condition.joblib
```

Optional `--engine-id`, `--session-id`, `--event-id`, `--start-utc`, and
`--end-utc` values must come from the acquisition system. Without them the
adapter leaves provenance null and warns that it cannot synchronize this
recording with other sensors. It maps only the dataset's two bearing-condition
classes to the envelope's `fault_prediction` and `fault_probabilities` fields;
the values in `fault_probabilities` are uncalibrated model scores, not verified
fault probabilities. Irregular timestamp quality is
marked poor, so the existing fusion layer will not use this output as
corroborating anomaly evidence.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_vibration_bearing.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The novelty framework remains separate and unlabeled. Its scores are novelty
alerts, not verified bearing-fault diagnoses.
