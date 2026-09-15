"""Shortcut-control benchmarks for weak target-noise learning."""

from __future__ import annotations

import numpy as np

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error for target-noise estimates."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination with a constant-target guard."""
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total <= 1e-15:
        return 0.0
    residual = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - residual / total


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation, returning 0 when variance vanishes."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Fit a small linear probe and report how much variance it explains."""
    x_design = np.column_stack([np.ones(x.shape[0]), x])
    weights = np.linalg.lstsq(x_design, y, rcond=None)[0]
    pred = x_design @ weights
    return r2_score(y, pred)


def shortcut_control_report(
    target_true: np.ndarray,
    target_pred: np.ndarray,
    depolarizing_rate: np.ndarray,
    measurement_rate: np.ndarray,
    seed: int = 0,
) -> dict[str, float]:
    """Measure whether predictions track target signal more than nuisance rates."""
    rng = np.random.default_rng(seed)
    permuted_target = rng.permutation(target_true)
    nuisance = np.column_stack([depolarizing_rate, measurement_rate, depolarizing_rate + measurement_rate])

    return {
        "shortcut/target_mae": mae(target_true, target_pred),
        "shortcut/random_pairing_baseline_mae": mae(permuted_target, target_pred),
        "shortcut/pred_target_corr": _safe_corr(target_true, target_pred),
        "shortcut/pred_depolarizing_corr": _safe_corr(depolarizing_rate, target_pred),
        "shortcut/pred_measurement_corr": _safe_corr(measurement_rate, target_pred),
        "shortcut/nuisance_linear_r2_on_prediction": _linear_r2(nuisance, target_pred),
    }
