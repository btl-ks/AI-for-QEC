"""Repetition code: the smallest stabilizer code the decoding pipeline accepts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_qec.qec.codes.stabilizer import StabilizerCode


@dataclass(frozen=True)
class RepetitionCode(StabilizerCode):
    """``d`` data qubits in a line, protected against one error type.

    Neighbouring pairs are compared, so check ``i`` reads ``e_i + e_{i+1}``.  A chain
    has an all-zero syndrome exactly when it is constant, so the only closed chains
    are the empty one and the all-ones chain, and the code carries one logical bit
    of distance ``d``.
    """

    name: str = "repetition_code"
    distance: int = 5

    def __post_init__(self) -> None:
        # Reject this state when self.distance < 2.
        if self.distance < 2:
            raise ValueError("Repetition-code distance must be at least 2")

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

    def parity_check_matrix(self) -> np.ndarray:
        """Return the ``(d-1) x d`` neighbour-comparison checks."""
        checks = np.zeros((self.distance - 1, self.distance), dtype=np.uint8)
        rows = np.arange(self.distance - 1)
        checks[rows, rows] = 1
        checks[rows, rows + 1] = 1
        return checks

    def logical_readout_matrix(self) -> np.ndarray:
        """Return a single-qubit readout.

        The only non-trivial closed chain flips every qubit, so reading any one
        qubit already separates the two logical classes.
        """
        readout = np.zeros((1, self.distance), dtype=np.uint8)
        readout[0, 0] = 1
        return readout

    def trivial_cycle_generators(self) -> np.ndarray:
        """Return no generators: the conjugate stabilizer group of this code is trivial.

        The repetition code corrects a single error type, so the empty chain is the
        only closed chain in the trivial class and no generator can be listed.
        """
        return np.zeros((0, self.distance), dtype=np.uint8)
