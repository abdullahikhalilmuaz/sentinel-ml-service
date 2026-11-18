import pytest

from src.metrics import evaluate_authentication


def test_far_frr_and_detection_metrics():
    # Three impostor attempts: two rejected, one accepted.
    # Three genuine attempts: one rejected, two accepted.
    result = evaluate_authentication(
        y_true_is_impostor=[1, 1, 1, 0, 0, 0],
        y_pred_rejected=[1, 1, 0, 1, 0, 0],
    )
    assert result["far"] == pytest.approx(1 / 3)
    assert result["frr"] == pytest.approx(1 / 3)
    assert result["detectionRate"] == pytest.approx(2 / 3)
    assert result["counts"]["impostorsAccepted"] == 1
    assert result["counts"]["genuineRejected"] == 1


def test_undefined_rates_are_none_not_fabricated_zero():
    result = evaluate_authentication([0, 0], [0, 1])
    assert result["far"] is None
    assert result["frr"] == 0.5


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        evaluate_authentication([0, 1], [1])
