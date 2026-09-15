"""Physical sampler for ideal toric-code code-capacity experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from ai_qec.qec.codes.toric_code import ToricCode
from ai_qec.qec.noise.phase_flip import IndependentPhaseFlipNoise
from ai_qec.qec.simulators.base import QECBackend


class ToricCodeCapacityBackend(QECBackend):
    """Sample real Bernoulli error chains and exact perfect-measurement syndromes."""

    name = "toric_code_capacity"

    def __init__(self, code: ToricCode, noise: IndependentPhaseFlipNoise) -> None:
        self.code = code
        self.noise = noise

    def sample_batch(self, n_samples: int, rng: np.random.Generator) -> dict[str, Any]:
        errors = self.noise.sample_errors(n_samples, self.code.num_data_qubits, rng)
        return {
            "physical_error": errors,
            "syndrome": self.code.syndrome(errors),
        }
