"""Balanced sampling utilities."""

from __future__ import annotations

import numpy as np


def target_bins(target_strength: np.ndarray) -> np.ndarray:
    """Map each discrete target strength to a stable integer bin."""
    unique = {value: idx for idx, value in enumerate(sorted(set(target_strength.tolist())))}
    return np.asarray([unique[value] for value in target_strength], dtype=np.int64)
