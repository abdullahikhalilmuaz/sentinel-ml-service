import pytest

from src.feature_schema import FEATURE_NAMES
from src.inference import score_personalized_baseline


@pytest.fixture
def feature_row():
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


def make_baseline(features):
    return {
        "status": "ready",
        "sampleCount": 30,
        "featureStats": {
            "mean": dict(features),
            "std": {name: 10.0 for name in FEATURE_NAMES},
        },
    }


def test_profile_match_has_zero_or_low_deviation(feature_row):
    score, signals = score_personalized_baseline(feature_row, make_baseline(feature_row))
    assert score == 0.0
    assert signals == []


def test_large_profile_deviation_increases_score(feature_row):
    baseline = make_baseline(feature_row)
    changed = dict(feature_row)
    changed["typingIntervalMean"] = 700.0
    changed["pointerSpeedMean"] = 1200.0
    changed["idleRatio"] = 0.8

    score, signals = score_personalized_baseline(changed, baseline)
    assert score > 0.7
    assert len(signals) >= 1
