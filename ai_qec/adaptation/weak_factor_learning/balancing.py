"""Nuisance balancing utilities."""

from __future__ import annotations

import numpy as np


def nuisance_score(depolarizing_rate: np.ndarray, measurement_rate: np.ndarray) -> np.ndarray:
    """Combine nuisance rates into one scalar bucket/sorting score."""
    return depolarizing_rate + measurement_rate
