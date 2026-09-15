"""Regression losses."""

from __future__ import annotations

import numpy as np


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """Compute Huber loss for robust target-regression training."""
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return float(np.mean(0.5 * quadratic**2 + delta * linear))
