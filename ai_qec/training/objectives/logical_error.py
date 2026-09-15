"""QEC-aware objective helpers."""

from __future__ import annotations

import numpy as np


def logical_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the fraction of incorrect logical predictions."""
    return float(np.mean(y_true != y_pred))
