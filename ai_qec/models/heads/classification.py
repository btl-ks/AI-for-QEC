"""Classification metrics."""

from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute thresholded binary classification metrics."""
    y_pred = (y_prob >= 0.5).astype(np.int64)
    accuracy = float(np.mean(y_pred == y_true))
    return {
        f"{prefix}/logical_accuracy": accuracy,
        f"{prefix}/logical_error_rate": 1.0 - accuracy,
        f"{prefix}/mean_logical_probability": float(np.mean(y_prob)),
    }
