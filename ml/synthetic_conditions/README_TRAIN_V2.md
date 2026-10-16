# V2 baseline training and inference

`train_v2.py` trains the separate `synthetic-conditions-167-v2-baseline-v1`
bundle from `ml/datasets/synthetic_conditions_v2/training/`. Before fitting, it
checks the generated handoff manifest, feature hash, source blob hashes,
serialized run evidence, exact 167-feature schema, finite values, five labels,
30 independent runs and 39 windows per run. It then evaluates the fixed V1
candidates: balanced 300-tree Random Forest and `StandardScaler + RBF SVC`.

Every fold is five-fold `GroupKFold` by `run_id`; run IDs are disjoint and each
fold contains all classes. The SVM scaler is fit inside each training fold.
After selection by mean fold macro-F1, the chosen estimator is fit once on all
validated rows and saved separately as
`ml/saved_models/synthetic_conditions_167_v2_baseline.joblib`. The report is
`ml/evaluation/synthetic_conditions_167_v2_groupkfold.json` and includes the
feature CSV hash, model hash, per-fold and out-of-fold accuracy, balanced
accuracy, macro-F1, confusion matrices and per-class recall. Existing V1 and
Fusion V1 artifacts are never loaded for fitting or overwritten.

```powershell
& "$env:TEMP\EngineGuardAI-ml-136-py314\Scripts\python.exe" -B `
  -m ml.synthetic_conditions.train_v2
```

The generated V2 result is descriptive alongside V1. Both currently use the
same synthetic condition generator family and both score perfectly on their
own grouped folds; this is not a controlled ablation of the added MAP/load/CKP
features and does not demonstrate improvement.

`predict_v2.py` accepts exactly the 167 finite numeric predictor names in the
manifest, in any JSON key order. Metadata, raw waveforms, missing values,
quality flags and strings are rejected. The model bundle requires all inputs
to have passed the V2 good-quality training contract. Output scores are
uncalibrated RF vote fractions or SVM decision scores. Every output is a
synthetic simulator-condition classification, not a real-engine diagnosis.
