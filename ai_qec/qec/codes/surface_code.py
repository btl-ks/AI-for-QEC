"""Lightweight surface-code metadata."""

from __future__ import annotations

from dataclasses import dataclass

from ai_qec.qec.codes.base import QECCode


@dataclass(frozen=True)
class SurfaceCode(QECCode):
    """Minimal rotated-surface-code metadata for synthetic experiments."""
    name: str = "surface_code"
    distance: int = 5

    @property
    def num_data_qubits(self) -> int:
        """Return the d² data qubits of a rotated surface code."""
        return self.distance * self.distance

    @property
    def num_stabilizers_per_round(self) -> int:
        """Return d²−1 stabilizer measurements per extraction round."""
        return self.distance * self.distance - 1

    @property
    def num_measurement_qubits(self) -> int:
        """The toy rotated-code layout uses one ancilla per stabilizer."""
        return self.num_stabilizers_per_round
