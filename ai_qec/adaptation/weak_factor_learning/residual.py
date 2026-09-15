"""Residual weak-signal helpers."""

from __future__ import annotations

import numpy as np


def residualize(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Subtract a nuisance or baseline estimate from observed values."""
    return values - baseline
