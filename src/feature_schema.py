"""Feature schema and strict validation shared by training and inference.

Feature units for schema version 1:
- typing/click intervals and their standard deviations: milliseconds
- pointer speed and its standard deviation: pixels per second
- idleRatio: fraction from 0.0 to 1.0
- navigationCount and interactionCount: non-negative counts per observation window

The frontend's feature extractor must use these same meanings and units.
"""
from __future__ import annotations

import math
from typing import Any

FEATURE_SCHEMA_VERSION = 1

FEATURE_NAMES = [
    "typingIntervalMean",
    "typingIntervalStd",
    "typingIntervalMedian",
    "pointerSpeedMean",
    "pointerSpeedStd",
    "clickIntervalMean",
    "clickIntervalStd",
    "idleRatio",
    "navigationCount",
    "interactionCount",
]

# Broad sanity limits, not behavioural decision thresholds.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "typingIntervalMean": (0.0, 60_000.0),
    "typingIntervalStd": (0.0, 60_000.0),
    "typingIntervalMedian": (0.0, 60_000.0),
    "pointerSpeedMean": (0.0, 1_000_000.0),
    "pointerSpeedStd": (0.0, 1_000_000.0),
    "clickIntervalMean": (0.0, 60_000.0),
    "clickIntervalStd": (0.0, 60_000.0),
    "idleRatio": (0.0, 1.0),
    "navigationCount": (0.0, 100_000.0),
    "interactionCount": (0.0, 100_000.0),
}
COUNT_FEATURES = {"navigationCount", "interactionCount"}


def feature_validation_error(features: Any) -> str | None:
    """Return a safe validation message, or None when features are valid."""
    if not isinstance(features, dict):
        return "features must be a JSON object"

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        return f"Missing required feature(s): {', '.join(missing)}"

    unknown = sorted(set(features) - set(FEATURE_NAMES))
    if unknown:
        return f"Unknown feature(s): {', '.join(unknown)}"

    for name in FEATURE_NAMES:
        value = features[name]
        # bool is a subclass of int in Python; deliberately reject it.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"Feature '{name}' must be a number"
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return f"Feature '{name}' must be finite"
        lower, upper = FEATURE_BOUNDS[name]
        if not lower <= numeric_value <= upper:
            return f"Feature '{name}' must be between {lower:g} and {upper:g}"
        if name in COUNT_FEATURES and not numeric_value.is_integer():
            return f"Feature '{name}' must be a whole number"

    return None


def validate_features(features: Any) -> bool:
    return feature_validation_error(features) is None


def baseline_validation_error(baseline: Any, minimum_samples: int = 20) -> str | None:
    """Validate the per-user profile contract used for personalized scoring.

    Expected object:
      {
        "status": "ready",
        "sampleCount": 20,
        "featureStats": {"mean": {feature: number, ...},
                         "std": {feature: number, ...}}
      }
    """
    if not isinstance(baseline, dict):
        return "baseline must be a JSON object"
    if baseline.get("status") != "ready":
        return "baseline.status must be 'ready'"

    sample_count = baseline.get("sampleCount")
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, float)):
        return "baseline.sampleCount must be a number"
    if not math.isfinite(float(sample_count)) or not float(sample_count).is_integer():
        return "baseline.sampleCount must be a finite whole number"
    if sample_count < minimum_samples:
        return f"baseline requires at least {minimum_samples} samples"

    stats = baseline.get("featureStats")
    if not isinstance(stats, dict):
        return "baseline.featureStats must be an object"

    for statistic_name in ("mean", "std"):
        values = stats.get(statistic_name)
        if not isinstance(values, dict):
            return f"baseline.featureStats.{statistic_name} must be an object"
        missing = [name for name in FEATURE_NAMES if name not in values]
        if missing:
            return f"Baseline {statistic_name} is missing: {', '.join(missing)}"
        unknown = sorted(set(values) - set(FEATURE_NAMES))
        if unknown:
            return f"Baseline {statistic_name} contains unknown feature(s): {', '.join(unknown)}"
        for name in FEATURE_NAMES:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"Baseline {statistic_name} for '{name}' must be a number"
            if not math.isfinite(float(value)):
                return f"Baseline {statistic_name} for '{name}' must be finite"
            if statistic_name == "std" and value < 0:
                return f"Baseline std for '{name}' cannot be negative"

    return None
