"""Common numpy layers."""

from __future__ import annotations

import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    """Apply the rectified linear activation elementwise."""
    return np.maximum(values, 0.0)
