"""Contract every stabilizer code must satisfy, checked without code-specific knowledge."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_qec.benchmarks.decoding.exact_posterior import closed_cycle_basis
from ai_qec.qec.codes.repetition_code import RepetitionCode
from ai_qec.qec.codes.stabilizer import StabilizerCode, gf2_matvec
from ai_qec.qec.codes.surface_code import SurfaceCode
from ai_qec.qec.codes.toric_code import ToricCode


CODES = [
    ToricCode(distance=3), ToricCode(distance=4),
    SurfaceCode(distance=3), SurfaceCode(distance=5),
    RepetitionCode(distance=3), RepetitionCode(distance=6),
]


def gf2_rank(matrix: np.ndarray) -> int:
    """Return the GF(2) rank of a binary matrix."""
    reduced = matrix.copy() % 2
    rank = 0
    for column in range(reduced.shape[1]):
        pivot = next((i for i in range(rank, len(reduced)) if reduced[i, column]), None)
        if pivot is None:
            continue
        reduced[[rank, pivot]] = reduced[[pivot, rank]]
        for i in range(len(reduced)):
            if i != rank and reduced[i, column]:
                reduced[i] ^= reduced[rank]
        rank += 1
    return rank


def all_closed_cycles(code: StabilizerCode) -> np.ndarray:
    """Enumerate ``ker H`` in full; only called on codes small enough to allow it."""
    basis = closed_cycle_basis(code)
    index = np.arange(1 << len(basis), dtype=np.uint64)
    bits = ((index[:, None] >> np.arange(len(basis), dtype=np.uint64)) & 1).astype(np.uint8)
    return (bits @ basis) % 2


class EveryCodeSatisfiesTheStabilizerContract(unittest.TestCase):
    """These hold for any code supplying ``H`` and ``L``, with no per-code branching."""

    def test_matrices_are_binary_and_dimensionally_consistent(self) -> None:
        for code in CODES:
            with self.subTest(code=code.name, distance=code.distance):
                check = code.parity_check_matrix()
                readout = code.logical_readout_matrix()
                self.assertTrue(np.isin(check, (0, 1)).all())
                self.assertTrue(np.isin(readout, (0, 1)).all())
                self.assertEqual(check.shape[1], code.num_data_qubits)
                self.assertEqual(readout.shape[1], code.num_data_qubits)
                self.assertEqual(check.shape[0], code.num_syndrome_bits)
                self.assertEqual(readout.shape[0], code.num_logical_bits)

    def test_syndrome_is_the_parity_check_product(self) -> None:
        for code in CODES:
            with self.subTest(code=code.name, distance=code.distance):
                rng = np.random.default_rng(code.distance)
                errors = (rng.random((200, code.num_data_qubits)) < 0.3).astype(np.uint8)
                expected = gf2_matvec(errors, code.parity_check_matrix())
                self.assertTrue(np.array_equal(code.syndrome(errors), expected))

    def test_readout_annihilates_the_trivial_cycle_generators(self) -> None:
        for code in CODES:
            with self.subTest(code=code.name, distance=code.distance):
                code.check_readout_is_well_defined()

    def test_logical_class_is_additive_and_trivial_on_stabilizers(self) -> None:
        for code in CODES:
            with self.subTest(code=code.name, distance=code.distance):
                cycles = all_closed_cycles(code)
                self.assertTrue(np.array_equal(
                    code.logical_class(cycles[1:] ^ cycles[:1]),
                    code.logical_class(cycles[1:]) ^ code.logical_class(cycles[:1]),
                ))
                generators = code.trivial_cycle_generators()
                if len(generators):
                    self.assertFalse(code.logical_class(generators).any())

    def test_logical_classes_partition_the_kernel_evenly(self) -> None:
        for code in CODES:
            with self.subTest(code=code.name, distance=code.distance):
                classes = code.logical_class(all_closed_cycles(code))
                labels, counts = np.unique(classes, axis=0, return_counts=True)
                self.assertEqual(len(labels), 2 ** code.num_logical_bits)
                self.assertEqual(len(set(counts.tolist())), 1)

    def test_code_distance_equals_the_declared_distance(self) -> None:
        for code in CODES:
            with self.subTest(code=code.name, distance=code.distance):
                cycles = all_closed_cycles(code)
                nontrivial = cycles[code.logical_class(cycles).any(axis=-1)]
                self.assertEqual(int(nontrivial.sum(axis=1).min()), code.distance)

    def test_logical_failure_is_driven_only_by_the_residual_class(self) -> None:
        for code in CODES:
            with self.subTest(code=code.name, distance=code.distance):
                rng = np.random.default_rng(code.distance + 50)
                cycles = all_closed_cycles(code)
                picked = cycles[rng.integers(0, len(cycles), 200)]
                errors = (rng.random((200, code.num_data_qubits)) < 0.1).astype(np.uint8)
                self.assertTrue(np.array_equal(
                    code.logical_failure(errors, errors ^ picked),
                    code.logical_class(picked).any(axis=-1),
                ))

    def test_kernel_dimension_matches_the_parity_check_rank(self) -> None:
        for code in CODES:
            with self.subTest(code=code.name, distance=code.distance):
                check = code.parity_check_matrix()
                free = code.num_data_qubits - gf2_rank(check)
                self.assertEqual(len(closed_cycle_basis(code)), free)


class RotatedSurfaceCodeGeometry(unittest.TestCase):
    """The two check types must form a valid CSS pair."""

    def test_check_types_commute_and_split_the_stabilizer_count(self) -> None:
        for distance in (3, 5, 7):
            with self.subTest(distance=distance):
                code = SurfaceCode(distance=distance)
                x_checks = code.parity_check_matrix()
                z_checks = code.trivial_cycle_generators()
                self.assertFalse(gf2_matvec(x_checks, z_checks).any())
                self.assertEqual(len(x_checks), (distance ** 2 - 1) // 2)
                self.assertEqual(len(z_checks), (distance ** 2 - 1) // 2)
                self.assertEqual(len(x_checks) + len(z_checks), code.num_stabilizers_per_round)

    def test_one_logical_qubit_remains(self) -> None:
        for distance in (3, 5, 7):
            with self.subTest(distance=distance):
                code = SurfaceCode(distance=distance)
                encoded = code.num_data_qubits - gf2_rank(code.parity_check_matrix()) \
                    - gf2_rank(code.trivial_cycle_generators())
                self.assertEqual(encoded, 1)

    def test_even_distance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SurfaceCode(distance=4)


class RepetitionCodeStructure(unittest.TestCase):
    """The only non-trivial closed chain flips every qubit."""

    def test_kernel_is_the_constant_chains(self) -> None:
        for distance in (3, 4, 6):
            with self.subTest(distance=distance):
                code = RepetitionCode(distance=distance)
                cycles = all_closed_cycles(code)
                self.assertEqual(len(cycles), 2)
                weights = sorted(cycles.sum(axis=1).tolist())
                self.assertEqual(weights, [0, distance])


if __name__ == "__main__":
    unittest.main()
