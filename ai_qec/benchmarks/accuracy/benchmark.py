"""Accuracy benchmark helpers."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute exact-match accuracy for label predictions."""
    return float(np.mean(y_true == y_pred))
