"""Common QEC circuit interface."""

from __future__ import annotations

from dataclasses import dataclass

from ai_qec.qec.codes.base import QECCode


@dataclass(frozen=True)
class QECCircuit:
    """Minimal circuit metadata shared by simulators and manifests."""
    name: str
    code: QECCode
    rounds: int

    @property
    def num_detectors(self) -> int:
        """Toy detector-event slots across all rounds, not circuit detectors.

        The value is explicitly scoped to the toy generator.  A real backend
        must derive detector count from its compiled circuit instead.
        """
        return self.rounds * self.code.num_stabilizers_per_round

    def context(self) -> dict[str, int | str]:
        """Return JSON-ready metadata for run and dataset manifests."""
        return {
            "circuit": self.name,
            "rounds": self.rounds,
            "toy_num_event_slots": self.num_detectors,
            **self.code.context(),
        }
