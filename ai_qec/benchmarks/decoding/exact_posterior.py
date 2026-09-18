"""Exact homology-class posteriors for small toric lattices.

For an ``L x L`` toric code the chains compatible with a syndrome form the coset
``e + ker H``, so the posterior over the four homology classes can be summed
exactly by enumerating ``ker H``.  This is only tractable for small lattices:
``ker H`` has ``2^(2L^2 - (L^2 - 1))`` elements, i.e. 2^17 at L=4 but 2^37 at L=6.

Two model-free references follow from those posteriors:

* **exact maximum likelihood** -- recover in the most likely class,
* **ideal posterior sampling** -- draw a class from the posterior.

Maximum likelihood needs a tie rule: the toric code's distance-``L`` degeneracy
makes exact ties between the trivial class and a logical class common (6%-12% of
shots at L=4), and the three rules in :class:`MaximumLikelihoodTieRule` bracket
the achievable failure rate.  Weights are compared as exact integers, so the
answer does not depend on floating-point summation order.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

import numpy as np

from ai_qec.qec.codes.toric_code import ToricCode


MAX_ENUMERATED_CYCLE_BITS = 24


class MaximumLikelihoodTieRule(str, Enum):
    """How an exact maximum-likelihood decoder resolves equally likely classes."""

    RANDOM = "random"           # expected failure under a uniform draw among the tied classes
    TRIVIAL_WINS = "trivial"    # optimistic bound: always keep the trivial class
    TIES_FAIL = "ties_fail"     # pessimistic bound: a tie is counted as a failure


@dataclass(frozen=True)
class ExactPosteriorReference:
    """Model-free decoding references for one lattice and error probability."""

    p_error: float
    samples: int
    exact_ml_p_fail: dict[str, float]
    ideal_sampling_p_fail: float
    tie_shots: int

    @property
    def tie_rate(self) -> float:
        return self.tie_shots / self.samples


def closed_cycle_basis(code: ToricCode) -> np.ndarray:
    """Return a GF(2) basis of ``ker H``: the chains with an all-zero syndrome."""
    parity_check = code.parity_check_matrix().astype(np.uint8) % 2
    width = code.num_data_qubits
    reduced = parity_check.copy()
    pivots: list[int] = []
    row = 0
    for column in range(width):
        source = next((i for i in range(row, len(reduced)) if reduced[i, column]), None)
        if source is None:
            continue
        reduced[[row, source]] = reduced[[source, row]]
        for i in range(len(reduced)):
            if i != row and reduced[i, column]:
                reduced[i] ^= reduced[row]
        pivots.append(column)
        row += 1
    free = [c for c in range(width) if c not in pivots]
    basis = np.zeros((len(free), width), dtype=np.uint8)
    for j, free_column in enumerate(free):
        basis[j, free_column] = 1
        for i, pivot_column in enumerate(pivots):
            basis[j, pivot_column] = reduced[i, free_column]
    if np.any(parity_check.astype(np.int64) @ basis.T.astype(np.int64) % 2):
        raise RuntimeError("closed-cycle basis does not lie in the kernel of H")
    return basis


def enumerate_closed_cycles(code: ToricCode) -> tuple[np.ndarray, np.ndarray]:
    """Return every closed cycle packed into one integer each, plus its homology class.

    The class index is ``2 * winding_x + winding_y``, so index 0 is the trivial class.
    """
    basis = closed_cycle_basis(code)
    if len(basis) > MAX_ENUMERATED_CYCLE_BITS:
        raise ValueError(
            f"enumerating 2^{len(basis)} closed cycles is not tractable; "
            f"the limit is 2^{MAX_ENUMERATED_CYCLE_BITS} (L=4 needs 2^17, L=6 would need 2^37)"
        )
    index = np.arange(1 << len(basis), dtype=np.uint32)
    coefficients = ((index[:, None] >> np.arange(len(basis), dtype=np.uint32)) & 1).astype(np.uint8)
    cycles = (coefficients @ basis) % 2
    winding = code.homology(cycles)
    classes = (winding[:, 0] * 2 + winding[:, 1]).astype(np.int64)
    return pack_chains(cycles), classes


def pack_chains(chains: np.ndarray) -> np.ndarray:
    """Pack binary chains into one unsigned integer each, one bit per qubit."""
    chains = np.asarray(chains, dtype=np.uint8)
    if chains.shape[-1] > 64:
        raise ValueError("chains longer than 64 qubits cannot be packed into uint64")
    return (chains.astype(np.uint64) << np.arange(chains.shape[-1], dtype=np.uint64)).sum(axis=-1)


def _weights(packed: np.ndarray) -> np.ndarray:
    """Return the Hamming weight of every packed chain."""
    return np.unpackbits(packed.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def homology_class_posteriors(code: ToricCode, errors: np.ndarray, p_error: float) -> np.ndarray:
    """Return the exact posterior over the four homology classes for each error chain.

    Column ``c`` is the probability that a recovery drawn from ``p(r | S(e))`` leaves
    the cycle ``e + r`` in class ``c``; column 0 is the trivial class.
    """
    if not 0.0 < p_error < 1.0:
        raise ValueError("p_error must lie strictly between 0 and 1")
    cycles, classes = enumerate_closed_cycles(code)
    order = np.argsort(classes, kind="stable")
    cycles, classes = cycles[order], classes[order]
    bounds = np.searchsorted(classes, np.arange(5))
    ratio = Fraction(p_error).limit_denominator(10**6)
    numerator, denominator = ratio.numerator, ratio.denominator - ratio.numerator
    if denominator <= 0:
        raise ValueError("p_error must be below 1 to give a positive weight ratio")

    packed_errors = pack_chains(np.asarray(errors, dtype=np.uint8))
    posteriors = np.zeros((len(packed_errors), 4), dtype=np.float64)
    for row, error in enumerate(packed_errors):
        weight = _weights(cycles ^ np.uint64(error))
        totals = []
        for sector in range(4):
            histogram = np.bincount(weight[bounds[sector]:bounds[sector + 1]], minlength=code.num_data_qubits + 1)
            totals.append(sum(
                int(count) * numerator**w * denominator**(code.num_data_qubits - w)
                for w, count in enumerate(histogram) if count
            ))
        total = sum(totals)
        posteriors[row] = [Fraction(value, total) for value in totals]
    return posteriors


def exact_posterior_reference(code: ToricCode, errors: np.ndarray, p_error: float) -> ExactPosteriorReference:
    """Return exact maximum-likelihood and ideal-posterior-sampling failure rates.

    Both references are model-free: they depend only on the code, the error chains
    and ``p_error``, so they bound what any decoder for this noise model can reach.
    """
    cycles, classes = enumerate_closed_cycles(code)
    order = np.argsort(classes, kind="stable")
    cycles, classes = cycles[order], classes[order]
    bounds = np.searchsorted(classes, np.arange(5))
    ratio = Fraction(p_error).limit_denominator(10**6)
    numerator, denominator = ratio.numerator, ratio.denominator - ratio.numerator

    packed_errors = pack_chains(np.asarray(errors, dtype=np.uint8))
    trivial_wins = ties_fail = tie_shots = 0
    random_failures = Fraction(0)
    ideal_failures = 0.0
    for error in packed_errors:
        weight = _weights(cycles ^ np.uint64(error))
        totals = []
        for sector in range(4):
            histogram = np.bincount(weight[bounds[sector]:bounds[sector + 1]], minlength=code.num_data_qubits + 1)
            totals.append(sum(
                int(count) * numerator**w * denominator**(code.num_data_qubits - w)
                for w, count in enumerate(histogram) if count
            ))
        total = sum(totals)
        ideal_failures += 1.0 - totals[0] / total
        largest = max(totals)
        tied = totals.count(largest)
        if totals[0] != largest:
            trivial_wins += 1
            ties_fail += 1
            random_failures += 1
        elif tied > 1:
            tie_shots += 1
            ties_fail += 1
            random_failures += Fraction(tied - 1, tied)

    samples = len(packed_errors)
    return ExactPosteriorReference(
        p_error=float(p_error),
        samples=samples,
        exact_ml_p_fail={
            MaximumLikelihoodTieRule.RANDOM.value: float(random_failures) / samples,
            MaximumLikelihoodTieRule.TRIVIAL_WINS.value: trivial_wins / samples,
            MaximumLikelihoodTieRule.TIES_FAIL.value: ties_fail / samples,
        },
        ideal_sampling_p_fail=ideal_failures / samples,
        tie_shots=tie_shots,
    )
