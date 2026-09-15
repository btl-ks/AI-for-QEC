"""Simulator factory."""

from __future__ import annotations

from typing import Any

from ai_qec.qec.circuits.base import QECCircuit
from ai_qec.qec.noise.base import NoiseModel
from ai_qec.qec.simulators.base import QECBackend
from ai_qec.qec.simulators.custom_backend import ToySyntheticQECBackend
from ai_qec.qec.simulators.toric_backend import ToricCodeCapacityBackend
from ai_qec.qec.codes.toric_code import ToricCode
from ai_qec.qec.noise.phase_flip import IndependentPhaseFlipNoise


def build_backend(config: dict[str, Any], circuit: QECCircuit, noise: NoiseModel) -> QECBackend:
    """Build the configured simulator backend."""
    generator = str(config["data"]["generator"]).lower()
    if generator == "toy_synthetic":
        return ToySyntheticQECBackend(circuit=circuit, noise=noise, config=config)
    if generator == "toric_code_capacity":
        if not isinstance(circuit.code, ToricCode) or not isinstance(noise, IndependentPhaseFlipNoise):
            raise ValueError("toric_code_capacity requires ToricCode and IndependentPhaseFlipNoise")
        return ToricCodeCapacityBackend(code=circuit.code, noise=noise)
    if generator == "stim":
        raise NotImplementedError("generator 'stim' is not implemented; Phase 1 provides the real Stim backend")
    raise ValueError(f"Unknown data generator/backend: {generator}")
