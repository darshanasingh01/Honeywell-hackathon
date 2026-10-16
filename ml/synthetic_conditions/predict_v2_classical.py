"""Strict inference for the separate V2 SVM and softmax artifacts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .schema_v2 import FEATURE_COLUMNS, MANIFEST, SCHEMA_VERSION
from .train_v2_classical import DEFAULT_SOFTMAX_MODEL, DEFAULT_SVM_MODEL, SVM_VERSION, SOFTMAX_VERSION


def load_model(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        bundle = joblib.load(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"classical V2 model not found at {path}") from exc
    if (bundle.get("schema_version") != SCHEMA_VERSION
            or bundle.get("feature_columns") != list(FEATURE_COLUMNS)
            or bundle.get("class_labels") != list(MANIFEST["class_labels"])
            or bundle.get("model_version") not in (SVM_VERSION, SOFTMAX_VERSION)):
        raise ValueError("model bundle does not match the classical V2 contract")
    return bundle


def _validated_frame(values: Mapping[str, Any]) -> pd.DataFrame:
    if not isinstance(values, Mapping):
        raise ValueError("input must be an object mapping all 167 feature names to values")
    expected, supplied = set(FEATURE_COLUMNS), set(values)
    if expected != supplied:
        raise ValueError(f"feature contract mismatch; missing={sorted(expected-supplied)}, extra={sorted(supplied-expected)}")
    row = []
    for name in FEATURE_COLUMNS:
        value = values[name]
        if isinstance(value, (bool, np.bool_)) or isinstance(value, str):
            raise ValueError(f"feature {name} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"feature {name} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"feature {name} must be finite")
        row.append(number)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_one(bundle: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    frame = _validated_frame(values)
    estimator = bundle["model"]
    model_classes = [str(x) for x in estimator.classes_]
    target_classes = list(MANIFEST["class_labels"])
    if set(model_classes) != set(target_classes):
        raise ValueError("model classes do not match V2 labels")
    predicted = str(estimator.predict(frame)[0])
    if bundle["model_version"] == SVM_VERSION:
        raw = np.asarray(estimator.decision_function(frame)).reshape(-1)
        semantics = "svm_decision_score_not_probability"
    else:
        raw = np.asarray(estimator.predict_proba(frame)).reshape(-1)
        semantics = "softmax_model_probability_not_calibrated"
    scores_by_model_class = dict(zip(model_classes, (float(x) for x in raw)))
    return {
        "schema_version": SCHEMA_VERSION,
        "prediction_type": "synthetic_engine_condition_classification",
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
        "predicted_condition": predicted,
        "class_order": target_classes,
        "condition_scores": {label: scores_by_model_class[label] for label in target_classes},
        "score_semantics": semantics,
        "dataset_provenance": dict(bundle["training_provenance"]),
        "limitations": ["Trained and evaluated on synthetic simulator features only.",
            "This does not establish real-engine fault-detection accuracy.",
            "SVM scores are not probabilities; softmax probabilities are not necessarily calibrated."],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_json", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = predict_one(load_model(args.model),
                             json.loads(args.feature_json.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
