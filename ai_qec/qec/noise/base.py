"""Noise model interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseDraw:
    """Batch of sampled target and nuisance noise parameters."""
    target_strength: np.ndarray
    depolarizing_rate: np.ndarray
    measurement_rate: np.ndarray


class NoiseModel:
    """Base interface for noise samplers."""
    name = "base"

    def sample(self, n_samples: int, rng: np.random.Generator) -> NoiseDraw:
        """Sample a batch of noise parameters."""
        raise NotImplementedError
