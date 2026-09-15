"""Normalization utilities."""

from __future__ import annotations

import numpy as np


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature-wise mean and nonzero scale for normalization."""
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return mean, scale


def apply_standardizer(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply a previously fitted feature standardizer."""
    return (features - mean) / scale
