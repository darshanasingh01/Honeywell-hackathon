"""Return the separate, synthetic-only condition-classification JSON contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .train import META_COLUMNS


def load_model(model_path: str | Path) -> dict[str, Any]:
    """Load the trained artifact or raise an actionable missing-file error."""
    path = Path(model_path)
    try:
        return joblib.load(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Synthetic-condition model artifact not found: {path}. "
            "Train it from the repository root with `python -m ml.synthetic_conditions.train "
            "--features ..\\Honeywell-features\\data\\processed\\features.csv`."
        ) from exc


def make_prediction(
    *,
    model_name: str,
    model_version: str,
    predicted_condition: str,
    condition_scores: Mapping[str, float],
    score_semantics: str,
) -> dict[str, Any]:
    """Build and validate a response that makes synthetic provenance explicit."""
    if predicted_condition not in condition_scores:
        raise ValueError("predicted_condition must be present in condition_scores")
    if not condition_scores:
        raise ValueError("condition_scores cannot be empty")
    normalized_scores = {}
    for label, score in condition_scores.items():
        if not isinstance(label, str) or not label:
            raise ValueError("condition score labels must be non-empty strings")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"condition score for {label!r} must be finite numeric")
        normalized_scores[label] = float(score)
    return {
        "schema_version": "engineguard.synthetic-condition/v1",
        "prediction_type": "synthetic_engine_condition_classification",
        "model_name": model_name,
        "model_version": model_version,
        "predicted_condition": predicted_condition,
        "condition_scores": normalized_scores,
        "score_semantics": score_semantics,
        "dataset_provenance": {
            "data_kind": "synthetic_simulated_engine_conditions",
            "real_engine_measurements": False,
        },
        "limitations": [
            "This prediction is based only on synthetic simulator features.",
            "It is not a validated real-world engine fault diagnosis or fault probability.",
            "The class score is model-specific and is not calibrated across models.",
        ],
    }


def predict_one(bundle: dict[str, Any], feature_values: Mapping[str, Any]) -> dict[str, Any]:
    """Predict from exactly one feature-only row; metadata fields are rejected."""
    if not isinstance(feature_values, Mapping):
        raise ValueError("input must be a mapping of numerical feature names to values")
    present_meta = set(feature_values) & set(META_COLUMNS)
    if present_meta:
        raise ValueError(f"metadata must not be supplied as model inputs: {sorted(present_meta)}")
    columns = list(bundle["feature_columns"])
    missing = set(columns) - set(feature_values)
    extra = set(feature_values) - set(columns)
    if missing or extra:
        raise ValueError(f"feature columns mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    row = pd.DataFrame([[feature_values[name] for name in columns]], columns=columns, dtype=float)
    if not np.isfinite(row.to_numpy()).all():
        raise ValueError("feature values must be finite")

    estimator = bundle["model"]
    predicted = str(estimator.predict(row)[0])
    if hasattr(estimator, "predict_proba"):
        raw = estimator.predict_proba(row)[0]
        classes = [str(label) for label in estimator.classes_]
        semantics = "random_forest_vote_fraction_uncalibrated"
    else:
        raw_scores = np.asarray(estimator.decision_function(row))
        if raw_scores.ndim == 1:
            raw_scores = raw_scores.reshape(1, -1)
        raw = raw_scores[0]
        final_estimator = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
        classes = [str(label) for label in final_estimator.classes_]
        semantics = "svm_decision_score_not_probability"
    scores = {label: float(score) for label, score in zip(classes, raw)}
    return make_prediction(
        model_name=str(bundle["model_name"]),
        model_version=str(bundle["model_version"]),
        predicted_condition=predicted,
        condition_scores=scores,
        score_semantics=semantics,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_json", type=Path, help="JSON object containing exactly the 74 feature values")
    parser.add_argument("--model", type=Path, default=Path("ml/saved_models/synthetic_conditions_baseline.joblib"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        bundle = load_model(args.model)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    feature_values = json.loads(args.feature_json.read_text(encoding="utf-8"))
    response = predict_one(bundle, feature_values)
    encoded = json.dumps(response, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
