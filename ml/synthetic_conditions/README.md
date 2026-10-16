# Synthetic engine-condition baselines

This isolated package evaluates Person B's simulator-derived feature table. It
does not alter the acoustic, vibration, CKP, or diagnostics/fusion models. Its
classes (`normal`, `knock`, `misfire`, `overheating`, `valve_train_wear`) are
**synthetic simulator labels**, not measurements or verified real-engine fault
labels. Do not merge these outputs into the validated sensor diagnostic
interface or describe them as real-world fault-detection accuracy.

## Input contract and leakage controls

The source is the read-only Person B handoff at
`Honeywell-features/data/processed/features.csv`, expected in a sibling worktree.
The expected table has 1170 rows: 74 numeric features plus the five metadata
columns `run_id`, `window_id`, `label`, `t_start`, and `t_end`. Model inputs
are formed by dropping all five metadata columns; `label` is the target and
`run_id` is used only as the group key. The feature loader checks numeric,
finite values, expected classes, and one label per run.

All evaluation uses `GroupKFold(n_splits=5, groups=run_id)`. The 50%-overlap
windows from a run therefore remain together. The `StandardScaler` is inside
the SVM pipeline and is fitted separately on each fold's training data. No
random window-level split, hyperparameter search, or full-dataset scaling is
used. Fold reports list train/test class coverage and held-out run IDs.

## Baselines

- Random Forest: 300 trees, `max_features="sqrt"`,
  `class_weight="balanced"`, seed 42, one worker.
- SVM: `StandardScaler` followed by RBF `SVC(C=1.0, gamma="scale",
  class_weight="balanced")`. SVM decision scores are not probabilities.

Selection is based on the highest mean grouped-fold macro-F1. The report also
contains each fold's accuracy, macro-F1, balanced accuracy, confusion matrix,
and per-class precision/recall/F1; it contains aggregate out-of-fold metrics
and confusion matrices. The selected estimator is refit on all 1170 rows only
after cross-validation and saved as a local Joblib artifact.

## Measured grouped-CV results

These are results from the validated Person B synthetic handoff, not real
engine performance. Class order in matrices is `normal`, `knock`, `misfire`,
`overheating`, `valve_train_wear`.

| Model | Fold | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Random Forest | 1 | 0.995726 | 0.994871 | 0.994872 |
| Random Forest | 2 | 0.991453 | 0.992403 | 0.994872 |
| Random Forest | 3 | 0.991453 | 0.992403 | 0.994872 |
| Random Forest | 4 | 0.995726 | 0.994871 | 0.994872 |
| Random Forest | 5 | 0.910256 | 0.883108 | 0.892308 |
| Scaled RBF SVM | 1 | 0.995726 | 0.994871 | 0.994872 |
| Scaled RBF SVM | 2 | 0.995726 | 0.996178 | 0.997436 |
| Scaled RBF SVM | 3 | 0.995726 | 0.996129 | 0.994872 |
| Scaled RBF SVM | 4 | 0.987179 | 0.984593 | 0.984615 |
| Scaled RBF SVM | 5 | 0.991453 | 0.989674 | 0.989744 |

| Model | Mean fold macro-F1 | OOF accuracy | OOF macro-F1 | OOF balanced accuracy |
|---|---:|---:|---:|---:|
| Random Forest | 0.971531 | 0.976923 | 0.976794 | 0.976923 |
| Scaled RBF SVM | **0.992289** | 0.993162 | **0.993169** | 0.993162 |

All five test folds contained all five classes; every fold trained on 24 runs
and tested on six disjoint runs. The scaled RBF SVM was selected by the stated
mean-fold macro-F1 rule. Its aggregate out-of-fold confusion matrix is:

```text
[[232,   0,   1,   1,   0],
 [  0, 234,   0,   0,   0],
 [  0,   0, 234,   0,   0],
 [  5,   0,   0, 229,   0],
 [  1,   0,   0,   0, 233]]
```

Its out-of-fold class precision / recall / F1 is: normal
0.974790 / 0.991453 / 0.983051; knock 1.000000 / 1.000000 / 1.000000;
misfire 0.995745 / 1.000000 / 0.997868; overheating 0.995652 / 0.978632 /
0.987069; valve_train_wear 1.000000 / 0.995726 / 0.997859. The complete
per-fold and per-class metrics for both models are in the JSON evaluation
report.

## Run and reproduce

First, regenerate Person B's artifacts from the `Honeywell-features` worktree
root using its existing environment:

```powershell
.\.venv\Scripts\python.exe -m feature_extraction.generate_dataset
.\.venv\Scripts\python.exe -m feature_extraction.pipeline
.\.venv\Scripts\python.exe -m feature_extraction.validate_b_outputs
```

Then, from this repository's root, using an environment with
`ml/requirements.txt` installed and the handoff worktree beside this one:

```powershell
.\.venv\Scripts\python.exe -m ml.synthetic_conditions.train `
  --features ..\Honeywell-features\data\processed\features.csv `
  --report ml\evaluation\synthetic_conditions_groupkfold.json `
  --model-output ml\saved_models\synthetic_conditions_baseline.joblib
```

The JSON report records the input file hash, software version, group folds,
metrics, class coverage, model parameters, and limitations. Dataset files,
NumPy arrays, and Joblib models remain local/ignored. The small evaluation
report is explicitly unignored for review.

The training CLI accepts any `--features` path, so adjust only that relative
path if the two worktrees are stored elsewhere. Absolute input paths are
reduced to their filename in the report to avoid machine-specific paths.

For a single feature-only JSON object with exactly the 74 feature fields:

```powershell
.\.venv\Scripts\python.exe -m ml.synthetic_conditions.predict `
  path\to\one_feature_row.json `
  --model ml\saved_models\synthetic_conditions_baseline.joblib
```

The response is a separate synthetic-condition JSON contract that a dashboard
can display as its own result panel. It labels the origin as simulated and
emits model-specific class scores (not calibrated fault probabilities). It is
not an input to the existing three-sensor diagnostics or fusion interfaces.
If the ignored model artifact is missing, the CLI prints a training command.

## Validation and limitations

Focused tests cover metadata exclusion, run-group separation, fold class
coverage, and the inference response contract:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_synthetic_conditions.py"
```

The group split measures generalization to held-out runs from the same
simulator and simulation design. It does not measure generalization to
independent engines, vehicles, real sensor hardware, or real faults. A strong
cross-validation score must be interpreted only within that synthetic scope.
