"""Periodic square-lattice toric code for code-capacity experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_qec.qec.codes.stabilizer import StabilizerCode, binary_vectors as _binary_edges


@dataclass(frozen=True)
class ToricCode(StabilizerCode):
    """Toric code with qubits on the links of an ``L x L`` periodic lattice.

    Horizontal edges are stored before vertical edges.  ``horizontal[y, x]``
    connects vertex ``(x, y)`` to ``(x+1, y)`` and ``vertical[y, x]``
    connects ``(x, y)`` to ``(x, y+1)``, in both cases modulo ``L``.
    """

    name: str = "toric_code"
    distance: int = 4

    def __post_init__(self) -> None:
        # Reject this state when self.distance < 2.
        if self.distance < 2:
            raise ValueError("Toric-code lattice size must be at least 2")

    @property
    def num_data_qubits(self) -> int:
        return 2 * self.distance * self.distance

    @property
    def num_stabilizers_per_round(self) -> int:
        """Return the X-vertex stabilizers relevant to Z-error decoding."""
        return self.distance * self.distance

    @property
    def num_measurement_qubits(self) -> int:
        """Report the vertex-check count for the idealized code-capacity model."""
        return self.num_stabilizers_per_round

    @property
    def num_syndrome_bits(self) -> int:
        return self.distance * self.distance

    def split_edges(self, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return horizontal and vertical edge tensors from a binary error chain."""
        error = _binary_edges(error, self.num_data_qubits, "error")
        shape = error.shape[:-1] + (self.distance, self.distance)
        cut = self.distance * self.distance
        return error[..., :cut].reshape(shape), error[..., cut:].reshape(shape)

    def syndrome(self, error: np.ndarray) -> np.ndarray:
        """Return X-vertex syndrome bits for one or more Z-error chains."""
        horizontal, vertical = self.split_edges(error)
        syndrome = (
            horizontal
            ^ np.roll(horizontal, 1, axis=-1)
            ^ vertical
            ^ np.roll(vertical, 1, axis=-2)
        )
        return syndrome.reshape(error.shape[:-1] + (self.num_syndrome_bits,))

    def parity_check_matrix(self) -> np.ndarray:
        """Return ``H`` with ``H @ error % 2 == syndrome(error)``; column ``j`` is the syndrome of edge ``j``."""
        return self.syndrome(np.eye(self.num_data_qubits, dtype=np.uint8)).T.copy()

    def logical_readout_matrix(self) -> np.ndarray:
        """Return the two dual cuts a cycle's winding parities count crossings of.

        Row 0 is the column of horizontal edges at ``x = L-1``, row 1 the row of
        vertical edges at ``y = L-1``.  A cycle crosses each cut an even number of
        times exactly when it does not wind that way, so these rows read the same
        two parities as :meth:`homology` while fitting the generic GF(2) form.
        """
        size = self.distance
        horizontal = np.zeros((size, size), dtype=np.uint8)
        vertical = np.zeros((size, size), dtype=np.uint8)
        horizontal[:, -1] = 1
        vertical[-1, :] = 1
        blank = np.zeros(size * size, dtype=np.uint8)
        return np.stack((
            np.concatenate((horizontal.ravel(), blank)),
            np.concatenate((blank, vertical.ravel())),
        ))

    def homology(self, cycle: np.ndarray, *, require_closed: bool = True) -> np.ndarray:
        """Return the two winding parities of a closed error/recovery cycle.

        This is :meth:`StabilizerCode.logical_class` under the toric code's own
        name, computed by reducing the cut edges directly instead of multiplying
        by the readout matrix; the unit tests pin the two to agree.
        """
        cycle = _binary_edges(cycle, self.num_data_qubits, "cycle")
        # Reject this state when require_closed and np.any(self.syndrome(cycle)).
        if require_closed and np.any(self.syndrome(cycle)):
            raise ValueError("Homology requires a closed cycle")
        horizontal, vertical = self.split_edges(cycle)
        winding_x = np.bitwise_xor.reduce(horizontal[..., :, -1], axis=-1)
        winding_y = np.bitwise_xor.reduce(vertical[..., -1, :], axis=-1)
        return np.stack((winding_x, winding_y), axis=-1).astype(np.uint8)

    def logical_class(self, cycle: np.ndarray, *, require_closed: bool = True) -> np.ndarray:
        """Return the homology class; the toric lattice reduces the cuts directly."""
        return self.homology(cycle, require_closed=require_closed)

    def trivial_cycle_generators(self) -> np.ndarray:
        """Return the plaquette chains that generate the contractible cycles.

        These are the Z-type face operators, not the X-vertex checks that make up
        ``H``: a face boundary is a closed chain, and adding one deforms a cycle
        without changing its homology class.
        """
        size = self.distance
        generators = []
        for y in range(size):
            for x in range(size):
                horizontal = np.zeros((size, size), dtype=np.uint8)
                vertical = np.zeros((size, size), dtype=np.uint8)
                horizontal[y, x] = horizontal[(y + 1) % size, x] = 1
                vertical[y, x] = vertical[y, (x + 1) % size] = 1
                generators.append(np.concatenate((horizontal.ravel(), vertical.ravel())))
        return np.stack(generators)
