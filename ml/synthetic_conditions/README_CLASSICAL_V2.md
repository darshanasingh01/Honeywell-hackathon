# V2 classical model extension

This extension trains separate synthetic-only baselines on the validated
167-feature V2 handoff using five-fold `GroupKFold` by `run_id`. Each fold fits
its own `StandardScaler`; no real-engine recordings are used.

Artifacts:

- `ml/saved_models/synthetic_conditions_167_v2_svm.joblib`
- `ml/saved_models/synthetic_conditions_167_v2_softmax.joblib`
- `ml/evaluation/synthetic_conditions_167_v2_classical_groupkfold.json`

The artifacts are Git-ignored and require separate delivery. Verify their
SHA-256 values against the evaluation report before use.

To load and predict from a JSON object containing exactly the 167 feature names:

```powershell
python -m ml.synthetic_conditions.predict_v2_classical `
  path\to\one_feature_row.json `
  --model ml/saved_models/synthetic_conditions_167_v2_svm.joblib

python -m ml.synthetic_conditions.predict_v2_classical `
  path\to\one_feature_row.json `
  --model ml/saved_models/synthetic_conditions_167_v2_softmax.joblib
```

SVM `condition_scores` are decision scores and are not probabilities.
Softmax `condition_scores` are model probabilities but are not necessarily
calibrated. Outputs use the schema label order:
`normal`, `knock`, `misfire`, `overheating`, `valve_train_wear`.

All metrics and predictions are synthetic-only and do not establish
real-engine fault-detection accuracy.
