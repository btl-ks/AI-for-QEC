"""Regularization utilities."""

from __future__ import annotations

import numpy as np


def l2_norm(weights: np.ndarray) -> float:
    """Compute squared L2 norm for regularization diagnostics."""
    return float(np.sum(weights**2))
