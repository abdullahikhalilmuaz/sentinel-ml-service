"""Evaluation helpers for labelled genuine/impostor test observations."""
from __future__ import annotations

from typing import Any, Iterable

from src.metrics import evaluate_authentication


def evaluate(
    y_true_is_impostor: Iterable[Any], y_pred_rejected: Iterable[Any]
) -> dict[str, Any]:
    """Evaluate predictions on a labelled holdout set.

    See src.metrics for the exact label convention and FAR/FRR definitions.
    Do not evaluate on the same data used to fit the model and call it a test set.
    """
    return evaluate_authentication(y_true_is_impostor, y_pred_rejected)
