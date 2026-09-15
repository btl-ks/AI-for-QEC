"""Idealized one-shot code-capacity experiment metadata."""

from __future__ import annotations

from ai_qec.qec.circuits.base import QECCircuit
from ai_qec.qec.codes.base import QECCode


class CodeCapacityExperiment(QECCircuit):
    """Perfect-syndrome experiment with physical data-qubit noise only."""

    def __init__(self, code: QECCode) -> None:
        super().__init__(name="code_capacity", code=code, rounds=1)

    def context(self) -> dict[str, int | str]:
        """Describe the ideal one-round syndrome measurement without toy fields."""
        return {
            "circuit": self.name,
            "rounds": self.rounds,
            "syndrome_bits": self.code.num_stabilizers_per_round,
            "measurement_fidelity": "perfect",
            "noise_scope": "data_qubits_only",
        }
