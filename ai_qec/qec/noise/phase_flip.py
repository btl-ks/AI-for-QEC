"""Independent Pauli-Z phase-flip noise for toric code-capacity decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_qec.qec.noise.base import NoiseDraw, NoiseModel


@dataclass(frozen=True)
class IndependentPhaseFlipNoise(NoiseModel):
    """Apply a Z error independently to each data qubit with probability ``p``."""

    p_error: float
    name: str = "phase_flip"

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_error <= 1.0:
            raise ValueError("p_error must lie in [0, 1]")

    def sample_errors(
        self,
        n_samples: int,
        num_data_qubits: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if n_samples < 1 or num_data_qubits < 1:
            raise ValueError("n_samples and num_data_qubits must be positive")
        return (rng.random((n_samples, num_data_qubits)) < self.p_error).astype(np.uint8)

    def sample(self, n_samples: int, rng: np.random.Generator) -> NoiseDraw:
        """Expose the fixed physical error probability through the common API."""
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        _ = rng
        return NoiseDraw(
            target_strength=np.full(n_samples, self.p_error, dtype=np.float64),
            depolarizing_rate=np.zeros(n_samples, dtype=np.float64),
            measurement_rate=np.zeros(n_samples, dtype=np.float64),
        )
