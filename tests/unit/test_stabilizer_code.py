"""Pin the toric code's hand-written fast paths to the generic GF(2) definitions."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_qec.benchmarks.decoding.exact_posterior import closed_cycle_basis
from ai_qec.qec.codes.stabilizer import StabilizerCode, gf2_matvec
from ai_qec.qec.codes.toric_code import ToricCode


LATTICE_SIZES = (2, 3, 4, 5, 6)


def random_closed_cycles(code: ToricCode, count: int, seed: int) -> np.ndarray:
    """Draw uniformly random elements of ``ker H`` so every logical class appears."""
    basis = closed_cycle_basis(code)
    rng = np.random.default_rng(seed)
    coefficients = (rng.random((count, len(basis))) < 0.5).astype(np.uint8)
    cycles = (coefficients @ basis) % 2
    assert not code.syndrome(cycles).any()
    return cycles


class ToricFastPathMatchesGenericDefinition(unittest.TestCase):
    """The lattice implementations are optimizations, never a second semantics."""

    def test_roll_syndrome_matches_parity_check_multiplication(self) -> None:
        for size in LATTICE_SIZES:
            with self.subTest(distance=size):
                code = ToricCode(distance=size)
                rng = np.random.default_rng(size)
                errors = (rng.random((500, code.num_data_qubits)) < 0.3).astype(np.uint8)
                generic = gf2_matvec(errors, code.parity_check_matrix())
                self.assertTrue(np.array_equal(code.syndrome(errors), generic))

    def test_winding_homology_matches_logical_readout_multiplication(self) -> None:
        for size in LATTICE_SIZES:
            with self.subTest(distance=size):
                code = ToricCode(distance=size)
                cycles = random_closed_cycles(code, 2000, seed=size)
                generic = StabilizerCode.logical_class(code, cycles)
                self.assertTrue(np.array_equal(code.homology(cycles), generic))

    def test_every_logical_class_is_reachable(self) -> None:
        for size in LATTICE_SIZES:
            with self.subTest(distance=size):
                code = ToricCode(distance=size)
                cycles = random_closed_cycles(code, 2000, seed=size + 100)
                classes = {tuple(row) for row in code.homology(cycles).tolist()}
                self.assertEqual(classes, {(0, 0), (0, 1), (1, 0), (1, 1)})


class LogicalReadoutIsWellDefined(unittest.TestCase):
    """A readout that does not annihilate the stabilizers would mislabel every class."""

    def test_stabilizer_generators_carry_the_trivial_class(self) -> None:
        for size in LATTICE_SIZES:
            with self.subTest(distance=size):
                code = ToricCode(distance=size)
                code.check_readout_is_well_defined(code.trivial_cycle_generators())

    def test_adding_a_stabilizer_never_changes_the_class(self) -> None:
        for size in LATTICE_SIZES:
            with self.subTest(distance=size):
                code = ToricCode(distance=size)
                cycles = random_closed_cycles(code, 500, seed=size + 200)
                generators = code.trivial_cycle_generators()
                rng = np.random.default_rng(size + 300)
                combination = (rng.random((500, len(generators))) < 0.5).astype(np.uint8)
                trivial = (combination @ generators) % 2
                self.assertTrue(np.array_equal(code.homology(cycles ^ trivial), code.homology(cycles)))

    def test_logical_class_is_additive(self) -> None:
        code = ToricCode(distance=4)
        left = random_closed_cycles(code, 400, seed=11)
        right = random_closed_cycles(code, 400, seed=12)
        self.assertTrue(np.array_equal(
            code.homology(left ^ right), code.homology(left) ^ code.homology(right)
        ))


class DerivedQuantitiesAgree(unittest.TestCase):
    """Closed-form counts must equal what the matrices report."""

    def test_qubit_and_syndrome_counts_match_the_parity_check_shape(self) -> None:
        for size in LATTICE_SIZES:
            with self.subTest(distance=size):
                code = ToricCode(distance=size)
                rows, columns = code.parity_check_matrix().shape
                self.assertEqual(code.num_data_qubits, columns)
                self.assertEqual(code.num_syndrome_bits, rows)
                self.assertEqual(code.num_logical_bits, 2)

    def test_logical_failure_matches_a_direct_class_check(self) -> None:
        code = ToricCode(distance=4)
        rng = np.random.default_rng(7)
        errors = (rng.random((300, code.num_data_qubits)) < 0.08).astype(np.uint8)
        cycles = random_closed_cycles(code, 300, seed=8)
        recoveries = errors ^ cycles
        self.assertTrue(np.array_equal(
            code.logical_failure(errors, recoveries), code.homology(cycles).any(axis=-1)
        ))

    def test_logical_failure_rejects_a_mismatched_recovery(self) -> None:
        code = ToricCode(distance=4)
        error = np.zeros(code.num_data_qubits, dtype=np.uint8)
        recovery = error.copy()
        recovery[0] = 1
        with self.assertRaises(ValueError):
            code.logical_failure(error[None, :], recovery[None, :])


if __name__ == "__main__":
    unittest.main()
