# Microphone acoustic models

HL-CEAD is the Humain-Lab Car Engine Audio Database published in the
[MachineLearningVisionRG repository](https://github.com/MachineLearningVisionRG/machine_biometrics).
It contains approximately 15-second, 44.1 kHz recordings at 1000, 1500, and
2000 RPM. Matching WAV stems across RPM folders identify the same car. The
recordings remain local and Git-ignored. The source repository does not state a
dataset license; do not redistribute the audio without permission.

## RPM classification

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe ml\acoustic\inspect_dataset.py ml\datasets\hl_cead\extracted\MLVRG-Car-Engine-Audio --class-depth 2
.\.venv\Scripts\python.exe ml\acoustic\train.py ml\datasets\hl_cead\extracted\MLVRG-Car-Engine-Audio
```

The classifier uses 3-second MFCC and spectral summary features, aggregated to
one vector per WAV, with vehicle-grouped stratified evaluation. RPM is a
metadata label, not a fault label.

Checkpoint 1 results: 5-fold vehicle-grouped out-of-fold accuracy is 73.68%
and macro-F1 is 0.7333 (152 recordings from 52 cars). The split keeps each
vehicle's RPM recordings together.

## RPM-conditioned acoustic novelty

```powershell
.\.venv\Scripts\python.exe ml\acoustic\novelty.py ml\datasets\hl_cead\extracted\MLVRG-Car-Engine-Audio
```

This fits a separate standardized Isolation Forest per RPM. Outer folds hold
vehicles out for evaluation; inner, vehicle-grouped cross-fitting calibrates
the per-RPM cutoff from training-only reference scores. The threshold is the
95th percentile, targeting a nominal 5% alert rate on calibration reference
scores. Held-out alert rates and score distributions are reported separately.
That nominal rate is not a false-fault rate. HL-CEAD has no verified
healthy/fault labels, so novelty means only deviation from this unlabeled
acoustic reference distribution.

The optional perturbations (impulsive knocks, periodic clicks, amplitude
modulation) are generated from held-out recordings and are always labelled
`SYNTHETIC PERTURBATIONS`. They are not authentic faults.

### Checkpoint 1 measured comparison

The baseline is selected for inference. Results below are vehicle-grouped
held-out novelty alert rates; perturbations are synthetic stress examples, not
faults:

| Evaluation set | Baseline | Transient experiment |
| --- | ---: | ---: |
| HL-CEAD reference recordings | 1.32% | 3.29% |
| Synthetic impulsive knocks | 1.32% | 3.29% |
| Synthetic periodic clicks | 84.21% | 76.32% |
| Synthetic amplitude modulation | 6.58% | 11.84% |

The transient features did not improve the selected knock-like perturbation
result and raised the reference alert rate, so the baseline remains selected.
Neither set of measurements establishes real fault-detection performance.
HL-CEAD has no verified healthy/fault labels; an alert means acoustic deviation
from its unlabeled reference distribution. Rates may change on new vehicles,
recording conditions, or authentic faults. Do not interpret alerts as a
diagnosis.

Default outputs:

- `ml/saved_models/acoustic_hlcead_novelty.joblib`
- `ml/evaluation/acoustic_hlcead_novelty.json`

The baseline model above is the inference default in `predict.py`. The
preserved transient experiment is stored locally as
`ml/saved_models/acoustic_hlcead_novelty_transient.joblib`; rerun it with
`--feature-set transient --model-out ml/saved_models/acoustic_hlcead_novelty_transient.joblib --report-out ml/evaluation/acoustic_hlcead_novelty_transient.json`.
Model binaries are Git-ignored; reproduce them with the commands in this file.

## Inference JSON

```powershell
.\.venv\Scripts\python.exe ml\acoustic\predict.py <audio.wav>
```

Inference emits `predicted_rpm`, `rpm_confidence`, `acoustic_novelty_score`,
`novelty_threshold`, `is_acoustically_unusual`, `fault_prediction`,
`model_version`, and `limitations`. RPM confidence is null unless the saved
classifier supports probabilities. Novelty fields are null until the novelty
artifact exists. `fault_prediction` remains null; no supervised fault model is
validated. Candidate dataset research is documented in
[DATASET_REVIEW.md](DATASET_REVIEW.md).
