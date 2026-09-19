"""Rotated surface code on a ``d x d`` patch of data qubits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_qec.qec.codes.stabilizer import StabilizerCode


@dataclass(frozen=True)
class SurfaceCode(StabilizerCode):
    """Rotated surface code with ``d²`` data qubits and ``d²-1`` stabilizers.

    Data qubit ``(row, column)`` is stored at index ``row * d + column``.  Faces of
    the ``d x d`` grid carry weight-four checks in a checkerboard, and each edge of
    the patch carries weight-two checks of the type whose logical string ends there.

    ``parity_check_matrix`` returns the X-type checks only, which are what detect the
    Z errors of a code-capacity experiment; the Z-type checks generate the chains
    that leave the logical class alone and are returned by
    :meth:`trivial_cycle_generators`.  Each type has ``(d²-1)/2`` checks, so the
    patch holds one logical qubit of distance ``d``.
    """

    name: str = "surface_code"
    distance: int = 5

    def __post_init__(self) -> None:
        # Reject distances that cannot define the supported rotated patch.
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("Rotated surface-code distance must be an odd integer of at least 3")

    @property
    def num_data_qubits(self) -> int:
        """Return the d² data qubits of a rotated surface code."""
        return self.distance * self.distance

    @property
    def num_stabilizers_per_round(self) -> int:
        """Return d²−1 stabilizer measurements per extraction round, counting both types."""
        return self.distance * self.distance - 1

    @property
    def num_measurement_qubits(self) -> int:
        """The toy rotated-code layout uses one ancilla per stabilizer."""
        return self.num_stabilizers_per_round

    def _checks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the X-type and Z-type check supports of the rotated patch."""
        size = self.distance
        x_checks: list[np.ndarray] = []
        z_checks: list[np.ndarray] = []

        def support(*qubits: tuple[int, int]) -> np.ndarray:
            row = np.zeros(size * size, dtype=np.uint8)
            for r, c in qubits:
                row[r * size + c] = 1
            return row

        for r in range(size - 1):
            for c in range(size - 1):
                face = support((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))
                # Assign even-parity faces to the X-check sublattice.
                if (r + c) % 2 == 0:
                    x_checks.append(face)
                # Assign odd-parity faces to the Z-check sublattice.
                else:
                    z_checks.append(face)
        for c in range(size - 1):
            # Place even-indexed horizontal boundary checks on the top edge.
            if c % 2 == 0:
                z_checks.append(support((0, c), (0, c + 1)))
            # Place odd-indexed horizontal boundary checks on the bottom edge.
            else:
                z_checks.append(support((size - 1, c), (size - 1, c + 1)))
        for r in range(size - 1):
            # Place odd-indexed vertical boundary checks on the left edge.
            if r % 2 == 1:
                x_checks.append(support((r, 0), (r + 1, 0)))
            # Place even-indexed vertical boundary checks on the right edge.
            else:
                x_checks.append(support((r, size - 1), (r + 1, size - 1)))
        return np.stack(x_checks), np.stack(z_checks)

    def parity_check_matrix(self) -> np.ndarray:
        """Return the X-type checks, whose outcomes are the syndrome of a Z-error chain."""
        return self._checks()[0]

    def logical_readout_matrix(self) -> np.ndarray:
        """Return one row supported on the top boundary row of data qubits.

        That row commutes with every Z-type check and meets the shortest non-trivial
        closed chain once, so its parity separates the two logical classes.
        """
        readout = np.zeros((1, self.num_data_qubits), dtype=np.uint8)
        readout[0, : self.distance] = 1
        return readout

    def trivial_cycle_generators(self) -> np.ndarray:
        """Return the Z-type checks, which generate the closed chains of trivial class."""
        return self._checks()[1]
