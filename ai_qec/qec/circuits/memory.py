"""Memory experiment circuit metadata."""

from __future__ import annotations

from ai_qec.qec.circuits.base import QECCircuit
from ai_qec.qec.codes.base import QECCode


class MemoryExperiment(QECCircuit):
    """Repeated-round memory experiment circuit."""

    def __init__(self, code: QECCode, rounds: int) -> None:
        """Create memory-experiment metadata for one code and round count."""
        super().__init__(name="memory", code=code, rounds=rounds)
