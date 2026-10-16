"""Read-only verification of the V2 GitHub handoff artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib

from .predict_v2 import FEATURE_COLUMNS, MANIFEST
from .predict_v2_classical import load_model as load_classical_model

ROOT = Path(__file__).resolve().parents[2]
LABELS = tuple(MANIFEST["class_labels"])
MODEL_HASHES = {
    "ml/saved_models/synthetic_conditions_167_v2_baseline.joblib": "01f693c753d3439f0eb4b7660aac4d95ab339529d4f5a53bc4435c2e11966db3",
    "ml/saved_models/synthetic_conditions_167_v2_svm.joblib": "b9cc549ed3e3fd516a3cf75614c0d3673e6f203fca30fb76027a8dbb96d11dba",
    "ml/saved_models/synthetic_conditions_167_v2_softmax.joblib": "1e418a050ba80346ae8d11cffb086ba20bad8dfbb0a2c4b5daec2c9a2d6c9edc",
}
SPECTRO_HASHES = {
    "normal": "cd8a34eb9e24021e7a7ba95016529b5f372a72947ee75b32d093c6d00ee7a23d",
    "knock": "4f76c5a2e35e31baa950313a4c9cd15bc56e7a9a6710ef60cb2601475f3e847c",
    "misfire": "26638705de53c9c17f820e1a018d2140ba8a56f54208c7931fcda1cd706cebbe",
    "overheating": "7eaabf3d0515d1bf4f571dcf7bee12635ebd4ec31bc8ef7acae61e05c50d7114",
    "valve_train_wear": "ad31ae6137ba5415cd1300c126619b167041c78be6689a757119b1993e15a6ae",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path = ROOT) -> dict:
    for relative, expected in MODEL_HASHES.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise AssertionError(f"model missing or hash mismatch: {relative}")
        if path.name.endswith("baseline.joblib"):
            bundle = joblib.load(path)
        else:
            bundle = load_classical_model(path)
        if tuple(bundle["feature_columns"]) != tuple(FEATURE_COLUMNS):
            raise AssertionError(f"feature schema mismatch: {relative}")
        if tuple(bundle["class_labels"]) != LABELS:
            raise AssertionError(f"label ordering mismatch: {relative}")

    sensor_index_path = root / "ml/synthetic_conditions/sensor_exports_v2_all_conditions/index.json"
    sensor_index = json.loads(sensor_index_path.read_text(encoding="utf-8"))
    sensor_records = dict(sensor_index["exports"])
    if set(sensor_records) != set(LABELS):
        raise AssertionError("sensor export condition coverage mismatch")
    for label, record in sensor_records.items():
        path = sensor_index_path.parent / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise AssertionError(f"sensor export missing or hash mismatch: {label}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload["condition_label"], payload["run_id"], payload["window_id"]) != (
            label, record["run_id"], record["window_id"]
        ):
            raise AssertionError(f"sensor window identity mismatch: {label}")

    spectro_index_path = root / "ml/synthetic_conditions/fast_spectrograms_v2_complete_retry/index.json"
    spectro_index = json.loads(spectro_index_path.read_text(encoding="utf-8"))
    spectro_records = dict(spectro_index["exports"])
    if set(spectro_records) != set(LABELS):
        raise AssertionError("spectrogram condition coverage mismatch")
    for label, record in spectro_records.items():
        path = spectro_index_path.parent / record["path"]
        if not path.is_file() or sha256(path) != SPECTRO_HASHES[label]:
            raise AssertionError(f"spectrogram missing or hash mismatch: {label}")
        sensor = sensor_records[label]
        if (record["run_id"], record["window_id"]) != (sensor["run_id"], sensor["window_id"]):
            raise AssertionError(f"spectrogram window identity mismatch: {label}")

    return {
        "status": "PASS",
        "models": list(MODEL_HASHES),
        "conditions": list(LABELS),
        "sensor_windows": len(sensor_records),
        "spectrogram_files": len(spectro_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
