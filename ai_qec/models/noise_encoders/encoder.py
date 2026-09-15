"""Simple learned-noise representation."""

from __future__ import annotations

import numpy as np

from ai_qec.models.noise_encoders.base import NoiseEncoder


class IdentityNoiseEncoder(NoiseEncoder):
    """No-op encoder that treats input features as the noise representation."""

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Return features unchanged."""
        return features
