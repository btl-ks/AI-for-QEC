"""Common code interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QECCode:
    """Minimal QEC code description used by generators and benchmarks."""

    name: str
    distance: int

    @property
    def num_data_qubits(self) -> int:
        """Return the number of physical data qubits in this code."""
        raise NotImplementedError

    @property
    def num_stabilizers_per_round(self) -> int:
        """Return stabilizer measurements made in one syndrome round.

        This is deliberately not called a detector count: circuit-level detector
        events depend on the compiled circuit and are introduced in Phase 1.
        """
        raise NotImplementedError

    @property
    def num_measurement_qubits(self) -> int:
        """Return ancilla/measurement qubits for one syndrome round."""
        raise NotImplementedError

    def context(self) -> dict[str, int | str]:
        """Return JSON-ready code metadata for manifests."""
        return {
            "code": self.name,
            "distance": self.distance,
            "num_data_qubits": self.num_data_qubits,
            "num_stabilizers_per_round": self.num_stabilizers_per_round,
            "num_measurement_qubits": self.num_measurement_qubits,
        }
