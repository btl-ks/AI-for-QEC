"""Importance sampling placeholders."""

from __future__ import annotations

import numpy as np


def uniform_weights(n_samples: int) -> np.ndarray:
    """Return normalized uniform importance weights."""
    return np.ones(n_samples, dtype=np.float64) / max(1, n_samples)
