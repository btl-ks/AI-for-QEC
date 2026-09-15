"""Noise encoder interface."""

from __future__ import annotations

import numpy as np


class NoiseEncoder:
    """Base interface for syndrome-to-noise representation encoders."""

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode model features into a noise representation."""
        raise NotImplementedError
