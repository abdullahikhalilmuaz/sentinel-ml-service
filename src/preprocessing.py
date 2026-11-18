"""Stable feature ordering for the Sentinel model."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.feature_schema import FEATURE_NAMES, validate_features


def preprocess(features: dict[str, Any]) -> np.ndarray:
    """Convert one validated feature object into a (1, 10) float64 matrix."""
    if not validate_features(features):
        raise ValueError("Cannot preprocess invalid behavioural features")
    return np.asarray([[float(features[name]) for name in FEATURE_NAMES]], dtype=np.float64)


def preprocess_rows(rows: Iterable[dict[str, Any]]) -> np.ndarray:
    """Convert a collection of feature objects into a (n, 10) matrix."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("At least one feature row is required")
    if any(not validate_features(row) for row in materialized):
        raise ValueError("Training data contains an invalid feature row")
    return np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in materialized],
        dtype=np.float64,
    )
