"""Combined model metrics."""

from __future__ import annotations

import numpy as np

from ai_qec.models.heads.classification import binary_metrics
from ai_qec.models.heads.regression import regression_metrics


def multitask_metrics(
    target_true: np.ndarray,
    target_pred: np.ndarray,
    logical_true: np.ndarray,
    logical_prob: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Combine regression and logical-decoding metrics under one prefix."""
    return {
        **regression_metrics(target_true, target_pred, prefix),
        **binary_metrics(logical_true, logical_prob, prefix),
    }
