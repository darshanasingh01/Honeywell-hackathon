"""Strict single-row inference for the separate synthetic 167-feature V2 model."""
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
from .train_v2 import DEFAULT_MODEL


def load_model(path: str | Path = DEFAULT_MODEL) -> dict[str, Any]:
    try:
        bundle = joblib.load(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"V2 model not found at {path}; run train_v2.py first") from exc
    if (bundle.get("schema_version") != SCHEMA_VERSION
            or bundle.get("feature_columns") != list(FEATURE_COLUMNS)
            or bundle.get("class_labels") != list(MANIFEST["class_labels"])
            or bundle.get("model_version") != "synthetic-conditions-167-v2-baseline-v1"):
        raise ValueError("model bundle does not match the V2 contract")
    return bundle


def predict_one(bundle: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError("input must be an object mapping all 167 feature names to values")
    expected, supplied = set(FEATURE_COLUMNS), set(values)
    if expected != supplied:
        raise ValueError(f"feature contract mismatch; missing={sorted(expected-supplied)}, extra={sorted(supplied-expected)}")
    row = []
    for name in FEATURE_COLUMNS:
        value = values[name]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"feature {name} must be numeric, not boolean")
        if isinstance(value, str):
            raise ValueError(f"feature {name} must be numeric, not a string")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"feature {name} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"feature {name} must be finite")
        row.append(number)
    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    estimator = bundle["model"]
    predicted = str(estimator.predict(frame)[0])
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(frame)[0]
        classes = [str(x) for x in estimator.classes_]
        semantics = "random_forest_vote_fraction_uncalibrated"
    else:
        scores = np.asarray(estimator.decision_function(frame)).reshape(-1)
        classifier = estimator.steps[-1][1]
        classes = [str(x) for x in classifier.classes_]
        semantics = "svm_decision_score_not_probability"
    return {"schema_version": "engineguard.synthetic-condition/167-v2",
        "prediction_type": "synthetic_engine_condition_classification",
        "model_name": bundle["model_name"], "model_version": bundle["model_version"],
        "predicted_condition": predicted,
        "condition_scores": {label: float(score) for label, score in zip(classes, scores)},
        "score_semantics": semantics,
        "dataset_provenance": dict(bundle["training_provenance"]),
        "limitations": ["Trained on synthetic simulator features only.",
            "This does not establish real-engine fault-detection accuracy.",
            "Scores are uncalibrated model-specific values, not fault probabilities."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_json", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = predict_one(load_model(args.model), json.loads(args.feature_json.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
