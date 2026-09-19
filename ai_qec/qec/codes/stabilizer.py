"""Code-agnostic decoding interface built from two binary matrices.

Everything the decoding pipeline needs from a code reduces to linear algebra over
GF(2).  A code supplies

* ``H`` -- the parity-check matrix, one row per stabilizer, so ``S = He``;
* ``L`` -- the logical readout matrix, one row per logical degree of freedom.

``L`` holds the qubit support of the logical operators of the type conjugate to
the errors being decoded, so ``Lc`` reads off which logical class a closed chain
``c`` belongs to.  For it to define a class at all, every stabilizer ``s`` must
satisfy ``Ls = 0``: adding a stabilizer moves a chain within its class, never
between classes.  Subclasses own those two matrices; syndromes, logical classes
and logical failure then follow for every stabilizer code without new code.

A code with a cheap closed form for a quantity (the toric lattice knows its qubit
count without building ``H``) should override the property.  A subclass whose
``parity_check_matrix`` is derived by pushing the identity through ``syndrome``
**must** override :attr:`num_data_qubits`, or the two will recurse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_qec.qec.codes.base import QECCode


def binary_vectors(value: np.ndarray, width: int, label: str) -> np.ndarray:
    """Validate one or more binary row vectors of the given width."""
    array = np.asarray(value, dtype=np.uint8)
    # Reject this state when array.ndim < 1 or array.shape[-1] != width.
    if array.ndim < 1 or array.shape[-1] != width:
        raise ValueError(f"{label} must end with dimension {width}, got {array.shape}")
    # Reject this state when not np.isin(array, (0, 1)).all().
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{label} must be binary")
    return array


def gf2_matvec(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return ``vectors @ matrix.T mod 2``, widened so long rows cannot overflow."""
    return (vectors.astype(np.int64) @ matrix.astype(np.int64).T % 2).astype(np.uint8)


@dataclass(frozen=True)
class StabilizerCode(QECCode):
    """A code whose decoding semantics follow from ``H`` and the logical readout ``L``."""

    def parity_check_matrix(self) -> np.ndarray:
        """Return ``H`` with ``H @ error % 2 == syndrome(error)``; one row per stabilizer."""
        raise NotImplementedError

    def logical_readout_matrix(self) -> np.ndarray:
        """Return ``L``; row ``i`` is the qubit support that reads logical class bit ``i``."""
        raise NotImplementedError

    @property
    def num_data_qubits(self) -> int:
        return int(self.parity_check_matrix().shape[1])

    @property
    def num_syndrome_bits(self) -> int:
        return int(self.parity_check_matrix().shape[0])

    @property
    def num_logical_bits(self) -> int:
        return int(self.logical_readout_matrix().shape[0])

    def syndrome(self, error: np.ndarray) -> np.ndarray:
        """Return the stabilizer measurement outcomes for one or more error chains."""
        error = binary_vectors(error, self.num_data_qubits, "error")
        return gf2_matvec(error, self.parity_check_matrix())

    def logical_class(self, cycle: np.ndarray, *, require_closed: bool = True) -> np.ndarray:
        """Return the logical class of one or more closed chains.

        The all-zero class is the trivial one, in which recovery succeeds.
        """
        cycle = binary_vectors(cycle, self.num_data_qubits, "cycle")
        # Reject this state when require_closed and np.any(self.syndrome(cycle)).
        if require_closed and np.any(self.syndrome(cycle)):
            raise ValueError("Logical class requires a closed cycle")
        return gf2_matvec(cycle, self.logical_readout_matrix())

    def logical_failure(self, error: np.ndarray, recovery: np.ndarray) -> np.ndarray:
        """Return whether a syndrome-compatible recovery leaves a non-trivial logical class."""
        error = binary_vectors(error, self.num_data_qubits, "error")
        recovery = binary_vectors(recovery, self.num_data_qubits, "recovery")
        # Reject this state when error.shape != recovery.shape.
        if error.shape != recovery.shape:
            raise ValueError("error and recovery shapes must match")
        # Reject this state when not np.array_equal(self.syndrome(error), self.syndrome(recovery)).
        if not np.array_equal(self.syndrome(error), self.syndrome(recovery)):
            raise ValueError("Recovery syndrome does not match the physical error")
        return np.any(self.logical_class(error ^ recovery), axis=-1)

    def trivial_cycle_generators(self) -> np.ndarray:
        """Return chains generating the closed cycles that carry the trivial class.

        These are the stabilizers of the type conjugate to ``H``: adding one moves a
        chain within its logical class.  A code whose conjugate stabilizer group is
        trivial returns an empty ``(0, n)`` array.
        """
        raise NotImplementedError

    def check_readout_is_well_defined(self, stabilizers: np.ndarray | None = None) -> None:
        """Raise unless ``L`` assigns every given stabilizer the trivial class.

        Callers pass a generating set; if any generator reads a non-zero class, the
        readout mixes stabilizers into logical information and every class is wrong.
        """
        # Follow this branch when stabilizers is None.
        if stabilizers is None:
            stabilizers = self.trivial_cycle_generators()
        stabilizers = binary_vectors(stabilizers, self.num_data_qubits, "stabilizers")
        # Return early when not len(stabilizers).
        if not len(stabilizers):
            return
        # Reject this state when np.any(self.syndrome(stabilizers)).
        if np.any(self.syndrome(stabilizers)):
            raise ValueError("stabilizers must be closed chains")
        # Reject this state when np.any(self.logical_class(stabilizers)).
        if np.any(self.logical_class(stabilizers)):
            raise ValueError("logical readout matrix does not annihilate the stabilizer group")
