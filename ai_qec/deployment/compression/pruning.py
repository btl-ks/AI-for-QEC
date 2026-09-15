"""Pruning placeholder."""

from __future__ import annotations

import numpy as np


def magnitude_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    """Return a boolean mask that keeps weights above a magnitude threshold."""
    return np.abs(values) >= threshold
