# V2 GitHub handoff

This handoff is synthetic-only. It contains the validated 167-feature inference
contract, five-condition sensor exports, and fast-sensor spectrogram evidence.
Real audio, real OBD telemetry, raw datasets, ZIP archives, and older experimental
models remain separate.

The three V2 model artifacts are 825,940 bytes (RF), 594,897 bytes (RBF SVM),
and 20,699 bytes (softmax). The five verified spectrogram NPZ files are 432,719
to 433,108 bytes each. The combined binary payload is about 3.4 MiB, so normal
Git tracking is suitable and Git LFS or a release asset is unnecessary. `*.joblib`
is ignored by the repository rules; add only the three named model files with
`git add -f`. Add only the five named NPZ files explicitly. Do not change
`.gitignore`.

## Fresh clone

```powershell
git clone <repository-url>
Set-Location Honeywell-hackathon
git checkout feature/ml-models
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r ml\requirements.txt
& .\.venv\Scripts\python.exe -B -m ml.synthetic_conditions.verify_v2_handoff
```

The verifier checks model hashes and schema/label metadata, five-condition
sensor coverage, relative export paths, hashes, and matching run/window IDs.
Prediction interfaces are `ml.synthetic_conditions.predict_v2` for RF and
`ml.synthetic_conditions.predict_v2_classical` for SVM/softmax. SVM decision
scores are not probabilities; softmax scores are probabilities from the fitted
classifier but are not guaranteed to be calibrated.

Loading and inference work from a fresh clone once these binaries are present.
Exact regeneration of the validated training artifacts is not promised by this
handoff because the source training CSV is intentionally excluded from Git.

The exact allowlist for delivery is the V2 Python sources, focused tests,
documentation, evaluation JSON, sensor JSON/index, spectrogram exporter/index,
the five spectrogram NPZ files, and the three V2 joblibs. Exclude `sensor_exports_v2.zip`,
audio NPZ/WAV files, raw datasets, training CSVs, unrelated reports, and the two
empty local directories left by aborted export attempts.
