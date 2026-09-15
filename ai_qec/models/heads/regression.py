"""Regression metrics."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination with a constant-target guard."""
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total <= 1e-15:
        return 0.0
    residual = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - residual / total


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    """Return MAE, RMSE, and R2 using a metric-name prefix."""
    return {
        f"{prefix}/target_mae": mae(y_true, y_pred),
        f"{prefix}/target_rmse": rmse(y_true, y_pred),
        f"{prefix}/target_r2": r2_score(y_true, y_pred),
    }
