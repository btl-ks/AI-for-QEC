"""Noise-estimation helper."""

from __future__ import annotations

import numpy as np


def estimate_target_from_predictions(predictions: np.ndarray) -> np.ndarray:
    """Treat model predictions as the current target-noise estimate."""
    return predictions
