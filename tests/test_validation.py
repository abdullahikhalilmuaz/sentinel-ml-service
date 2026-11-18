import math

import pytest

from src.feature_schema import (
    FEATURE_NAMES,
    baseline_validation_error,
    feature_validation_error,
    validate_features,
)


@pytest.fixture
def valid_features():
    return {
        "typingIntervalMean": 220.0,
        "typingIntervalStd": 65.0,
        "typingIntervalMedian": 215.0,
        "pointerSpeedMean": 350.0,
        "pointerSpeedStd": 120.0,
        "clickIntervalMean": 850.0,
        "clickIntervalStd": 300.0,
        "idleRatio": 0.2,
        "navigationCount": 2,
        "interactionCount": 28,
    }


def test_valid_feature_schema(valid_features):
    assert validate_features(valid_features)
    assert feature_validation_error(valid_features) is None


def test_missing_feature_is_rejected(valid_features):
    valid_features.pop("typingIntervalMean")
    assert not validate_features(valid_features)
    assert "Missing" in feature_validation_error(valid_features)


def test_extra_feature_is_rejected(valid_features):
    valid_features["password"] = "never collect this"
    assert not validate_features(valid_features)
    assert "Unknown" in feature_validation_error(valid_features)


@pytest.mark.parametrize("bad_value", [True, "100", None, float("nan"), float("inf"), -1])
def test_invalid_numeric_values_are_rejected(valid_features, bad_value):
    valid_features["typingIntervalMean"] = bad_value
    assert not validate_features(valid_features)


def test_idle_ratio_must_be_in_range(valid_features):
    valid_features["idleRatio"] = 1.01
    assert not validate_features(valid_features)


def test_counts_must_be_whole_numbers(valid_features):
    valid_features["navigationCount"] = 1.5
    assert not validate_features(valid_features)


def test_baseline_requires_ready_profile_and_enough_samples(valid_features):
    baseline = {
        "status": "ready",
        "sampleCount": 20,
        "featureStats": {
            "mean": dict(valid_features),
            "std": {name: 10.0 for name in FEATURE_NAMES},
        },
    }
    assert baseline_validation_error(baseline) is None
    baseline["sampleCount"] = 5
    assert "at least" in baseline_validation_error(baseline)
