"""Simple repetition-code baseline metadata."""

from __future__ import annotations

from dataclasses import dataclass

from ai_qec.qec.codes.base import QECCode


@dataclass(frozen=True)
class RepetitionCode(QECCode):
    """Minimal repetition-code metadata for baseline experiments."""
    name: str = "repetition_code"
    distance: int = 5

    @property
    def num_data_qubits(self) -> int:
        """Return the number of data qubits in the repetition code."""
        return self.distance

    @property
    def num_stabilizers_per_round(self) -> int:
        """Return the number of parity checks per round."""
        return max(1, self.distance - 1)

    @property
    def num_measurement_qubits(self) -> int:
        """Use one measurement qubit per parity check in the toy layout."""
        return self.num_stabilizers_per_round
