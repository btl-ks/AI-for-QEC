"""Classification losses."""

from __future__ import annotations

import numpy as np


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute numerically clipped binary cross entropy."""
    y_prob = np.clip(y_prob, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
