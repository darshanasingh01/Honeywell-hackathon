# EngineGuard ML workspace

This workspace holds the acoustic, vibration, and crankshaft position (CKP) model code. Dataset files and generated artifacts are kept out of Git.

## Windows PowerShell setup

Install Python 3.11 or 3.12 and make sure the Python launcher can find it (`py -0p`). From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r ml\requirements.txt
python ml\verify_environment.py
```

If PowerShell blocks activation, run the environment's interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r ml\requirements.txt
.\.venv\Scripts\python.exe ml\verify_environment.py
```

To leave the environment, run `deactivate` after activation. No dataset is downloaded or model trained by these setup steps.

## Workspace map

- `acoustic/`: dataset inspection, preprocessing, features, training, and prediction modules.
- `vibration/`: sampling-rate-aware signal features and grouped novelty baseline framework; dataset research is documented in `vibration/DATASET_REVIEW.md`.
- `ckp/`: crankshaft position timing model workspace.
- `datasets/`: local datasets (ignored by Git).
- `saved_models/`: locally generated model artifacts (ignored by Git).
- `evaluation/`: local evaluation outputs (generated outputs ignored by Git).

The microphone acoustic RPM and novelty pipeline is implemented under `acoustic/` (Checkpoint 1). No vibration dataset has been downloaded or trained in this workspace; vibration novelty outputs are not verified fault diagnoses. CKP remains a separate module.
