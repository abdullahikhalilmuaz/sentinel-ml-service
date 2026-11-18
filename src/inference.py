"""Model loading and anomaly scoring for Sentinel.

The population Isolation Forest is a demo novelty detector until trained on
representative, consented genuine-session data. It is not a probability and it
cannot identify a specific user without a personalized baseline.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from config import (
    ARTIFACT_FORMAT_VERSION,
    FEATURE_SCHEMA_VERSION,
    MODEL_PATH,
    MODEL_VERSION,
)
from src.feature_schema import FEATURE_NAMES
from src.preprocessing import preprocess


class ModelUnavailableError(RuntimeError):
    """Raised when there is no valid model artifact to score a batch."""


_cached_artifact: dict[str, Any] | None = None
_cached_mtime_ns: int | None = None
_cached_error: str | None = None


def _load_artifact_if_needed() -> dict[str, Any] | None:
    global _cached_artifact, _cached_mtime_ns, _cached_error

    path = Path(MODEL_PATH)
    if not path.is_file():
        _cached_artifact = None
        _cached_mtime_ns = None
        _cached_error = f"Model artifact not found at {path}. Run 'python -m src.train'."
        return None

    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        _cached_artifact = None
        _cached_mtime_ns = None
        _cached_error = "Model artifact could not be inspected"
        return None

    if _cached_mtime_ns == mtime_ns and (_cached_artifact is not None or _cached_error):
        return _cached_artifact

    try:
        artifact = joblib.load(path)
        if not isinstance(artifact, dict):
            raise ValueError("Unsupported artifact format; retrain the model")
        if artifact.get("artifactFormatVersion") != ARTIFACT_FORMAT_VERSION:
            raise ValueError("Unsupported artifact version; retrain the model")
        if artifact.get("schemaVersion") != FEATURE_SCHEMA_VERSION:
            raise ValueError("Artifact feature schema does not match this service")
        if artifact.get("featureNames") != FEATURE_NAMES:
            raise ValueError("Artifact feature order does not match this service")
        if not callable(getattr(artifact.get("model"), "decision_function", None)):
            raise ValueError("Artifact does not contain a compatible model")
        calibration = artifact.get("calibration")
        if not isinstance(calibration, dict):
            raise ValueError("Artifact is missing score calibration")
        lower = float(calibration["lowScoreQuantile01"])
        boundary = float(calibration["boundaryScoreQuantile05"])
        if not math.isfinite(lower) or not math.isfinite(boundary) or boundary <= lower:
            raise ValueError("Artifact score calibration is invalid")
        _cached_artifact = artifact
        _cached_mtime_ns = mtime_ns
        _cached_error = None
        return _cached_artifact
    except Exception as exc:  # The API returns a safe status; detail is for local logs/status only.
        _cached_artifact = None
        _cached_mtime_ns = mtime_ns
        _cached_error = str(exc)
        return None


def _get_artifact_or_raise() -> dict[str, Any]:
    artifact = _load_artifact_if_needed()
    if artifact is None:
        raise ModelUnavailableError(_cached_error or "Model artifact is unavailable")
    return artifact


def get_model_status() -> dict[str, Any]:
    artifact = _load_artifact_if_needed()
    if artifact is None:
        return {
            "status": "not_loaded",
            "modelAvailable": False,
            "modelVersion": None,
            "schemaVersion": FEATURE_SCHEMA_VERSION,
            "artifactPath": str(MODEL_PATH),
            "reason": _cached_error or "Model artifact is unavailable",
        }

    training_kind = str(artifact.get("trainingDataKind", "unknown"))
    return {
        "status": "loaded",
        "modelAvailable": True,
        "modelVersion": str(artifact.get("modelVersion", MODEL_VERSION)),
        "schemaVersion": FEATURE_SCHEMA_VERSION,
        "modelType": str(artifact.get("modelType", "IsolationForest")),
        "trainingDataKind": training_kind,
        "trainingSampleCount": int(artifact.get("trainingSampleCount", 0)),
        "trainedAt": artifact.get("trainedAt"),
        # Synthetic data is fine for plumbing/tests, not for claiming validated identity recognition.
        "evaluationStatus": str(artifact.get("evaluationStatus", "not_evaluated")),
        "productionReady": artifact.get("evaluationStatus") == "validated",
        "personalizedBaselineSupported": True,
    }


def _population_anomaly_score(artifact: dict[str, Any], features: dict[str, Any]) -> float:
    matrix = preprocess(features)
    decision_score = float(artifact["model"].decision_function(matrix)[0])
    calibration = artifact["calibration"]
    low = float(calibration["lowScoreQuantile01"])
    boundary = float(calibration["boundaryScoreQuantile05"])

    # Scores above the 5th-percentile boundary are inside the learned normal region.
    if decision_score >= boundary:
        return 0.0
    if decision_score <= low:
        return 1.0
    return float(np.clip((boundary - decision_score) / (boundary - low), 0.0, 1.0))


# Floors prevent near-zero observed standard deviations from causing unstable scores.
_PROFILE_SCALE_FLOORS = {
    "typingIntervalMean": 20.0,
    "typingIntervalStd": 15.0,
    "typingIntervalMedian": 20.0,
    "pointerSpeedMean": 25.0,
    "pointerSpeedStd": 20.0,
    "clickIntervalMean": 40.0,
    "clickIntervalStd": 25.0,
    "idleRatio": 0.05,
    "navigationCount": 1.0,
    "interactionCount": 2.0,
}


def score_personalized_baseline(
    features: dict[str, Any], baseline: dict[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    """Compute an interpretable deviation score against an enrolled user's profile.

    Score is a policy-oriented anomaly index in [0, 1], not a probability.
    `baseline` must have passed baseline_validation_error().
    """
    means = baseline["featureStats"]["mean"]
    stds = baseline["featureStats"]["std"]
    deviations: list[dict[str, Any]] = []

    for name in FEATURE_NAMES:
        floor = _PROFILE_SCALE_FLOORS[name]
        spread = max(float(stds[name]), floor)
        z_score = abs(float(features[name]) - float(means[name])) / spread
        deviations.append({
            "feature": name,
            "zScore": round(min(z_score, 100.0), 3),
            "direction": "above" if float(features[name]) >= float(means[name]) else "below",
        })

    largest = sorted(deviations, key=lambda item: item["zScore"], reverse=True)
    strongest = largest[:3]
    mean_top_deviation = sum(item["zScore"] for item in strongest) / len(strongest)
    # 2 scale units is the tolerance floor; 6 scale units reaches the maximum score.
    score = float(np.clip((mean_top_deviation - 2.0) / 4.0, 0.0, 1.0))
    signals = [item for item in strongest if item["zScore"] >= 2.0]
    return round(score, 4), signals


def score_features(
    features: dict[str, Any], baseline: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Score features and return the stable contract consumed by Node.js."""
    artifact = _get_artifact_or_raise()
    population_score = _population_anomaly_score(artifact, features)
    training_kind = str(artifact.get("trainingDataKind", "unknown"))
    version = str(artifact.get("modelVersion", MODEL_VERSION))

    if baseline is not None:
        personalized_score, signals = score_personalized_baseline(features, baseline)
        return {
            "status": "scored",
            "anomalyScore": personalized_score,
            "scoreMode": "personalized_baseline",
            "baselineUsed": True,
            "populationAnomalyScore": round(population_score, 4),
            "signals": signals,
            "modelVersion": version,
            "schemaVersion": FEATURE_SCHEMA_VERSION,
            "trainingDataKind": training_kind,
            "scoreIsProbability": False,
        }

    return {
        "status": "scored",
        "anomalyScore": round(population_score, 4),
        "scoreMode": "population_demo",
        "baselineUsed": False,
        "populationAnomalyScore": round(population_score, 4),
        "signals": [],
        "modelVersion": version,
        "schemaVersion": FEATURE_SCHEMA_VERSION,
        "trainingDataKind": training_kind,
        "scoreIsProbability": False,
        "limitation": "No enrolled user baseline was supplied; this score is not user-specific authentication.",
    }
