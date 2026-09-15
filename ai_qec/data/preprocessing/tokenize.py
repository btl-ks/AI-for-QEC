"""Tokenization helpers for detector-summary features."""

from __future__ import annotations

import numpy as np


def tokenize_detector_summary(features: np.ndarray) -> np.ndarray:
    """Return summary features as model-ready numeric tokens."""

    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    return features.astype(np.float64, copy=False)
