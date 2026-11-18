"""Evaluation metrics for labelled continuous-authentication attempts.

Label convention:
- y_true_is_impostor = 1 means the attempt was an impostor; 0 means genuine.
- y_pred_rejected = 1 means the system rejected/challenged the attempt; 0 means accepted.

FAR = impostor attempts accepted / all impostor attempts.
FRR = genuine attempts rejected / all genuine attempts.
Undefined rates are returned as None when their denominator is zero.
"""
from __future__ import annotations

from typing import Iterable, Any


def _binary_values(values: Iterable[Any], name: str) -> list[int]:
    converted: list[int] = []
    for value in values:
        if isinstance(value, bool):
            converted.append(int(value))
        elif value in (0, 1):
            converted.append(int(value))
        else:
            raise ValueError(f"{name} must contain only 0/1 values")
    return converted


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def evaluate_authentication(
    y_true_is_impostor: Iterable[Any], y_pred_rejected: Iterable[Any]
) -> dict[str, Any]:
    truth = _binary_values(y_true_is_impostor, "y_true_is_impostor")
    pred = _binary_values(y_pred_rejected, "y_pred_rejected")
    if not truth or len(truth) != len(pred):
        raise ValueError("Ground-truth and prediction arrays must be non-empty and equal in length")

    impostor_total = sum(truth)
    genuine_total = len(truth) - impostor_total
    impostors_rejected = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1)
    impostors_accepted = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 0)
    genuine_rejected = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 1)
    genuine_accepted = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 0)

    precision = _safe_rate(impostors_rejected, impostors_rejected + genuine_rejected)
    recall = _safe_rate(impostors_rejected, impostor_total)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 6)

    return {
        "sampleCount": len(truth),
        "counts": {
            "impostorAttempts": impostor_total,
            "genuineAttempts": genuine_total,
            "impostorsRejected": impostors_rejected,
            "impostorsAccepted": impostors_accepted,
            "genuineRejected": genuine_rejected,
            "genuineAccepted": genuine_accepted,
        },
        "far": _safe_rate(impostors_accepted, impostor_total),
        "frr": _safe_rate(genuine_rejected, genuine_total),
        "precision": precision,
        "recall": recall,
        "detectionRate": recall,
        "f1": f1,
        "accuracy": round((impostors_rejected + genuine_accepted) / len(truth), 6),
        "metricDefinitions": {
            "far": "impostor attempts accepted divided by all impostor attempts",
            "frr": "genuine attempts rejected divided by all genuine attempts",
            "positiveClass": "impostor attempts rejected",
        },
    }
