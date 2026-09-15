"""Synthetic weak crosstalk model."""

from __future__ import annotations

from typing import Any

import numpy as np

from ai_qec.qec.noise.base import NoiseDraw, NoiseModel


class WeakCrosstalkNoise(NoiseModel):
    """Sample target crosstalk independently from nuisance rates."""

    name = "crosstalk"

    def __init__(self, config: dict[str, Any]) -> None:
        """Read target crosstalk values and nuisance ranges from config."""
        noise_cfg = config.get("noise", {})
        target = noise_cfg.get("target", {})
        nuisance = noise_cfg.get("nuisance", {})

        self.target_values = np.asarray(target.get("values", [0.0]), dtype=np.float64)
        if self.target_values.ndim != 1 or self.target_values.size == 0:
            raise ValueError("noise.target.values must be a non-empty list")

        depol = nuisance.get("depolarizing", {})
        meas = nuisance.get("measurement", {})
        self.depolarizing_min = float(depol.get("min", 0.0))
        self.depolarizing_max = float(depol.get("max", self.depolarizing_min))
        self.measurement_min = float(meas.get("min", 0.0))
        self.measurement_max = float(meas.get("max", self.measurement_min))

    def sample(self, n_samples: int, rng: np.random.Generator) -> NoiseDraw:
        """Sample target crosstalk and independent nuisance rates."""
        target_idx = rng.integers(0, self.target_values.size, size=n_samples)
        target = self.target_values[target_idx]
        depol = rng.uniform(self.depolarizing_min, self.depolarizing_max, size=n_samples)
        meas = rng.uniform(self.measurement_min, self.measurement_max, size=n_samples)
        return NoiseDraw(
            target_strength=target.astype(np.float64),
            depolarizing_rate=depol.astype(np.float64),
            measurement_rate=meas.astype(np.float64),
        )
