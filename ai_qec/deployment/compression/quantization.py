"""Quantization placeholder."""

from __future__ import annotations

import numpy as np


def int8_quantize(values: np.ndarray) -> np.ndarray:
    """Quantize floating-point values to signed int8 using max-abs scaling."""
    scale = max(float(np.max(np.abs(values))), 1e-12) / 127.0
    return np.round(values / scale).astype(np.int8)
