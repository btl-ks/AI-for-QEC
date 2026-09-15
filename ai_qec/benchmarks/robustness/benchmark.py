"""Robustness evaluation across nuisance ranges."""

from __future__ import annotations

import numpy as np

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error for one nuisance bucket."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error for one nuisance bucket."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nuisance_bucket_report(
    target_true: np.ndarray,
    target_pred: np.ndarray,
    nuisance_score: np.ndarray,
) -> dict[str, float]:
    """Report target prediction quality across low/medium/high nuisance buckets."""
    quantiles = np.quantile(nuisance_score, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    labels = ["low", "medium", "high"]
    report: dict[str, float] = {}

    for idx, label in enumerate(labels):
        lo = quantiles[idx]
        hi = quantiles[idx + 1]
        if idx == len(labels) - 1:
            mask = (nuisance_score >= lo) & (nuisance_score <= hi)
        else:
            mask = (nuisance_score >= lo) & (nuisance_score < hi)
        if not np.any(mask):
            report[f"robustness/{label}_count"] = 0.0
            continue
        report[f"robustness/{label}_count"] = float(mask.sum())
        report[f"robustness/{label}_target_mae"] = mae(target_true[mask], target_pred[mask])
        report[f"robustness/{label}_target_rmse"] = rmse(target_true[mask], target_pred[mask])

    return report
