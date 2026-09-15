"""Periodic square-lattice toric code for code-capacity experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_qec.qec.codes.base import QECCode


def _binary_edges(value: np.ndarray, width: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.uint8)
    if array.ndim < 1 or array.shape[-1] != width:
        raise ValueError(f"{label} must end with dimension {width}, got {array.shape}")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{label} must be binary")
    return array


@dataclass(frozen=True)
class ToricCode(QECCode):
    """Toric code with qubits on the links of an ``L x L`` periodic lattice.

    Horizontal edges are stored before vertical edges.  ``horizontal[y, x]``
    connects vertex ``(x, y)`` to ``(x+1, y)`` and ``vertical[y, x]``
    connects ``(x, y)`` to ``(x, y+1)``, in both cases modulo ``L``.
    """

    name: str = "toric_code"
    distance: int = 4

    def __post_init__(self) -> None:
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

    def homology(self, cycle: np.ndarray, *, require_closed: bool = True) -> np.ndarray:
        """Return the two winding parities of a closed error/recovery cycle."""
        cycle = _binary_edges(cycle, self.num_data_qubits, "cycle")
        if require_closed and np.any(self.syndrome(cycle)):
            raise ValueError("Homology requires a closed cycle")
        horizontal, vertical = self.split_edges(cycle)
        winding_x = np.bitwise_xor.reduce(horizontal[..., :, -1], axis=-1)
        winding_y = np.bitwise_xor.reduce(vertical[..., -1, :], axis=-1)
        return np.stack((winding_x, winding_y), axis=-1).astype(np.uint8)

    def logical_failure(self, error: np.ndarray, recovery: np.ndarray) -> np.ndarray:
        """Return whether a syndrome-compatible recovery has nontrivial homology."""
        error = _binary_edges(error, self.num_data_qubits, "error")
        recovery = _binary_edges(recovery, self.num_data_qubits, "recovery")
        if error.shape != recovery.shape:
            raise ValueError("error and recovery shapes must match")
        if not np.array_equal(self.syndrome(error), self.syndrome(recovery)):
            raise ValueError("Recovery syndrome does not match the physical error")
        return np.any(self.homology(error ^ recovery), axis=-1)
