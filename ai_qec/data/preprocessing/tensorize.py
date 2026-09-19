"""Tensorization helpers."""

from __future__ import annotations

import numpy as np


def as_float_matrix(values: np.ndarray) -> np.ndarray:
    """Validate and expose a feature matrix as float64."""
    # Reject this state when values.ndim != 2.
    if values.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")
    return values.astype(np.float64, copy=False)
