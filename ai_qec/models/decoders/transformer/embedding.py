"""Feature embedding helpers."""

from __future__ import annotations

import numpy as np


def append_bias(features: np.ndarray) -> np.ndarray:
    """Prepend a constant bias column to a 2D feature matrix."""
    return np.column_stack([np.ones(features.shape[0], dtype=features.dtype), features])
