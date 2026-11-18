"""Train the Sentinel Isolation Forest artifact.

Default mode builds a clearly labelled synthetic DEMO model so the whole stack
can be wired and tested. For meaningful evaluation, use --csv with observed,
consented genuine-session feature rows matching schema version 1.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACT_FORMAT_VERSION,
    FEATURE_SCHEMA_VERSION,
    MODEL_PATH,
    MODEL_VERSION,
)
from src.feature_schema import FEATURE_NAMES, feature_validation_error
from src.preprocessing import preprocess_rows


def generate_synthetic_normal_samples(count: int = 5000, seed: int = 42) -> list[dict[str, Any]]:
    """Create plausible-valued data only for local wiring and demo purposes."""
    if count < 500:
        raise ValueError("Synthetic training requires at least 500 samples")

    rng = np.random.default_rng(seed)
    typing_mean = np.clip(rng.lognormal(mean=np.log(220.0), sigma=0.35, size=count), 50, 1500)
    typing_median = np.clip(typing_mean * rng.lognormal(mean=-0.02, sigma=0.10, size=count), 30, 1500)
    typing_std = np.clip(typing_mean * rng.uniform(0.12, 0.70, size=count), 5, 1000)

    pointer_mean = np.clip(rng.lognormal(mean=np.log(350.0), sigma=0.55, size=count), 5, 5000)
    pointer_std = np.clip(pointer_mean * rng.uniform(0.15, 1.10, size=count), 2, 8000)

    click_mean = np.clip(rng.lognormal(mean=np.log(850.0), sigma=0.55, size=count), 80, 8000)
    click_std = np.clip(click_mean * rng.uniform(0.12, 1.00, size=count), 5, 8000)

    idle_ratio = rng.beta(a=2.5, b=7.5, size=count)
    navigation_count = np.clip(rng.poisson(lam=2.5, size=count), 0, 1000).astype(int)
    interaction_count = np.clip(
        navigation_count + rng.poisson(lam=28.0, size=count), 0, 5000
    ).astype(int)

    rows: list[dict[str, Any]] = []
    for i in range(count):
        rows.append({
            "typingIntervalMean": float(typing_mean[i]),
            "typingIntervalStd": float(typing_std[i]),
            "typingIntervalMedian": float(typing_median[i]),
            "pointerSpeedMean": float(pointer_mean[i]),
            "pointerSpeedStd": float(pointer_std[i]),
            "clickIntervalMean": float(click_mean[i]),
            "clickIntervalStd": float(click_std[i]),
            "idleRatio": float(idle_ratio[i]),
            "navigationCount": int(navigation_count[i]),
            "interactionCount": int(interaction_count[i]),
        })
    return rows


def load_genuine_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Read feature-only CSV rows; caller must ensure rows represent genuine use."""
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Training CSV must have a header row")
        missing = [name for name in FEATURE_NAMES if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Training CSV is missing feature columns: {', '.join(missing)}")

        for line_number, raw_row in enumerate(reader, start=2):
            row: dict[str, Any] = {}
            for name in FEATURE_NAMES:
                try:
                    value = float(raw_row[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid number in row {line_number}, column '{name}'") from exc
                row[name] = int(value) if name in {"navigationCount", "interactionCount"} and value.is_integer() else value
            error = feature_validation_error(row)
            if error:
                raise ValueError(f"Invalid training data at row {line_number}: {error}")
            rows.append(row)

    if len(rows) < 100:
        raise ValueError("Provide at least 100 valid genuine-session rows for CSV training")
    return rows


def train_model(rows: list[dict[str, Any]], training_data_kind: str) -> dict[str, Any]:
    if len(rows) < 100:
        raise ValueError("At least 100 training rows are required")

    matrix = preprocess_rows(rows)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(
            n_estimators=300,
            contamination=0.05,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(matrix)
    normal_scores = np.asarray(model.decision_function(matrix), dtype=np.float64)
    low_score = float(np.quantile(normal_scores, 0.01))
    boundary_score = float(np.quantile(normal_scores, 0.05))
    if not np.isfinite(low_score) or not np.isfinite(boundary_score) or boundary_score <= low_score:
        raise RuntimeError("Could not calibrate model scores from the training set")

    artifact: dict[str, Any] = {
        "artifactFormatVersion": ARTIFACT_FORMAT_VERSION,
        "schemaVersion": FEATURE_SCHEMA_VERSION,
        "featureNames": list(FEATURE_NAMES),
        "modelVersion": MODEL_VERSION,
        "modelType": "IsolationForest",
        "model": model,
        "calibration": {
            "lowScoreQuantile01": low_score,
            "boundaryScoreQuantile05": boundary_score,
            "description": "Empirical decision-function quantiles from the training rows; output is an anomaly index, not a probability.",
        },
        "trainingDataKind": training_data_kind,
        "evaluationStatus": "not_evaluated",
        "trainingSampleCount": len(rows),
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "featureUnits": {
            "intervals": "milliseconds",
            "pointerSpeed": "pixels_per_second",
            "idleRatio": "fraction_0_to_1",
            "counts": "per_observation_window",
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Sentinel behavioural anomaly model")
    parser.add_argument(
        "--csv",
        help="Optional CSV containing schema-v1 genuine-session feature rows. Without it, trains a synthetic demo model.",
    )
    parser.add_argument("--samples", type=int, default=5000, help="Synthetic rows (minimum 500; default 5000)")
    args = parser.parse_args()

    if args.csv:
        rows = load_genuine_csv(args.csv)
        kind = "observed_genuine"
    else:
        rows = generate_synthetic_normal_samples(args.samples)
        kind = "synthetic_demo"

    artifact = train_model(rows, kind)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Model version: {artifact['modelVersion']}")
    print(f"Training data kind: {artifact['trainingDataKind']}")
    print(f"Training samples: {artifact['trainingSampleCount']}")
    if kind == "synthetic_demo":
        print("WARNING: This synthetic model is for local integration/demo only, not validated identity recognition.")


if __name__ == "__main__":
    main()
