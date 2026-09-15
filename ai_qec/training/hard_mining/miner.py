"""Hard example miner."""

from __future__ import annotations

import numpy as np


def mine(errors: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """Select the highest-error examples for hard-example mining."""
    k = max(1, int(errors.size * fraction))
    return np.argsort(errors)[-k:]
