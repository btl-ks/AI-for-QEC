"""Hard-example mining helpers."""

from __future__ import annotations

import numpy as np


def top_error_indices(errors: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k largest per-sample errors."""
    k = min(k, errors.size)
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    return np.argsort(errors)[-k:]
