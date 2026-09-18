"""Contracts for the model-free decoding references of the Torlai--Melko summary."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_qec.benchmarks.decoding.exact_posterior import (  # noqa: E402
    MaximumLikelihoodTieRule,
    closed_cycle_basis,
    enumerate_closed_cycles,
    exact_posterior_reference,
    homology_class_posteriors,
    pack_chains,
)
from ai_qec.benchmarks.decoding.toric import mwpm_tie_sensitivity  # noqa: E402
from ai_qec.qec.codes.toric_code import ToricCode  # noqa: E402


def gf2_rank(matrix: np.ndarray) -> int:
    """Return the GF(2) rank of a binary matrix."""
    reduced = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rank = 0
    for column in range(reduced.shape[1]):
        source = next((i for i in range(rank, len(reduced)) if reduced[i, column]), None)
        if source is None:
            continue
        reduced[[rank, source]] = reduced[[source, rank]]
        for i in range(len(reduced)):
            if i != rank and reduced[i, column]:
                reduced[i] ^= reduced[rank]
        rank += 1
    return rank


def brute_force_class_weights(code: ToricCode, error: np.ndarray, p_error: float) -> list[Fraction]:
    """Return exact per-homology-class weights by enumerating every chain of the code.

    This walks all ``2^n`` chains and keeps the syndrome-compatible ones, rather than
    enumerating ``ker H`` and translating by the error, so it shares no structure with
    the implementation under test beyond the rational conversion of ``p_error``.
    """
    ratio = Fraction(p_error).limit_denominator(10**6)
    target = code.syndrome(error)
    weights = [Fraction(0)] * 4
    for bits in itertools.product((0, 1), repeat=code.num_data_qubits):
        recovery = np.array(bits, dtype=np.uint8)
        if not np.array_equal(code.syndrome(recovery), target):
            continue
        support = int(recovery.sum())
        likelihood = ratio**support * (1 - ratio) ** (code.num_data_qubits - support)
        winding = code.homology(error ^ recovery)
        weights[int(winding[0]) * 2 + int(winding[1])] += likelihood
    return weights


class ClosedCycleEnumerationTest(unittest.TestCase):
    def test_basis_spans_the_kernel_of_the_parity_check(self) -> None:
        for distance in (2, 3, 4):
            with self.subTest(distance=distance):
                code = ToricCode(distance=distance)
                basis = closed_cycle_basis(code)
                # H has one redundant row: the product of all vertex checks is the identity.
                expected_dimension = code.num_data_qubits - (distance * distance - 1)
                self.assertEqual(basis.shape, (expected_dimension, code.num_data_qubits))
                self.assertEqual(gf2_rank(basis), expected_dimension)
                self.assertFalse(np.any(code.syndrome(basis)))

    def test_enumerated_cycles_are_closed_and_split_evenly_across_homology_classes(self) -> None:
        code = ToricCode(distance=2)
        cycles, classes = enumerate_closed_cycles(code)
        self.assertEqual(cycles.shape, (2**5,))
        # The four homology classes are cosets of the stabilizer group, so they are equal in size.
        np.testing.assert_array_equal(np.bincount(classes, minlength=4), [8, 8, 8, 8])
        trivial = cycles[classes == 0]
        unpacked = np.stack([
            (np.uint64(packed) >> np.arange(code.num_data_qubits, dtype=np.uint64)) & np.uint64(1)
            for packed in trivial
        ]).astype(np.uint8)
        self.assertFalse(np.any(code.homology(unpacked)))

    def test_enumeration_refuses_intractable_lattices(self) -> None:
        with self.assertRaises(ValueError) as raised:
            enumerate_closed_cycles(ToricCode(distance=6))
        self.assertIn("2^37", str(raised.exception))

    def test_pack_chains_puts_qubit_j_in_bit_j_and_rejects_wide_chains(self) -> None:
        code = ToricCode(distance=2)
        packed = pack_chains(np.eye(code.num_data_qubits, dtype=np.uint8))
        np.testing.assert_array_equal(packed, 1 << np.arange(code.num_data_qubits, dtype=np.uint64))
        self.assertEqual(int(pack_chains(np.ones(code.num_data_qubits, dtype=np.uint8))), 2**8 - 1)
        with self.assertRaises(ValueError):
            pack_chains(np.zeros((1, 65), dtype=np.uint8))


class HomologyClassPosteriorTest(unittest.TestCase):
    def test_posterior_matches_brute_force_enumeration_of_every_chain(self) -> None:
        code = ToricCode(distance=2)
        rng = np.random.default_rng(7)
        errors = (rng.random((6, code.num_data_qubits)) < 0.25).astype(np.uint8)
        posteriors = homology_class_posteriors(code, errors, 0.15)
        for row, error in enumerate(errors):
            with self.subTest(row=row):
                weights = brute_force_class_weights(code, error, 0.15)
                total = sum(weights)
                expected = [float(weight / total) for weight in weights]
                np.testing.assert_allclose(posteriors[row], expected, rtol=0, atol=1e-12)

    def test_posteriors_are_normalised_and_reject_degenerate_error_rates(self) -> None:
        code = ToricCode(distance=4)
        rng = np.random.default_rng(3)
        errors = (rng.random((4, code.num_data_qubits)) < 0.10).astype(np.uint8)
        posteriors = homology_class_posteriors(code, errors, 0.10)
        self.assertEqual(posteriors.shape, (4, 4))
        np.testing.assert_allclose(posteriors.sum(axis=1), np.ones(4), rtol=0, atol=1e-12)
        self.assertTrue(np.all(posteriors >= 0.0))
        for invalid in (0.0, 1.0, -0.1, 1.5):
            with self.subTest(p_error=invalid), self.assertRaises(ValueError):
                homology_class_posteriors(code, errors, invalid)


class ExactPosteriorReferenceTest(unittest.TestCase):
    def test_ideal_sampling_rate_is_the_mean_nontrivial_posterior_mass(self) -> None:
        code = ToricCode(distance=2)
        rng = np.random.default_rng(11)
        errors = (rng.random((16, code.num_data_qubits)) < 0.2).astype(np.uint8)
        reference = exact_posterior_reference(code, errors, 0.15)
        posteriors = homology_class_posteriors(code, errors, 0.15)
        self.assertEqual(reference.samples, len(errors))
        self.assertAlmostEqual(reference.ideal_sampling_p_fail, float((1.0 - posteriors[:, 0]).mean()), places=12)

    def test_tie_rules_bracket_the_achievable_failure_rate(self) -> None:
        code = ToricCode(distance=2)
        rng = np.random.default_rng(11)
        errors = (rng.random((16, code.num_data_qubits)) < 0.2).astype(np.uint8)
        reference = exact_posterior_reference(code, errors, 0.15)
        rates = reference.exact_ml_p_fail
        self.assertEqual(set(rates), {rule.value for rule in MaximumLikelihoodTieRule})
        optimistic = rates[MaximumLikelihoodTieRule.TRIVIAL_WINS.value]
        pessimistic = rates[MaximumLikelihoodTieRule.TIES_FAIL.value]
        self.assertLessEqual(optimistic, rates[MaximumLikelihoodTieRule.RANDOM.value])
        self.assertLessEqual(rates[MaximumLikelihoodTieRule.RANDOM.value], pessimistic)
        # The bracket is exactly the tied shots: the two bounds differ only on those.
        self.assertGreater(reference.tie_shots, 0)
        self.assertAlmostEqual(pessimistic - optimistic, reference.tie_rate, places=12)

    def test_ties_are_resolved_in_exact_arithmetic_rather_than_floating_point(self) -> None:
        code = ToricCode(distance=2)
        rng = np.random.default_rng(11)
        errors = (rng.random((16, code.num_data_qubits)) < 0.2).astype(np.uint8)
        # 0.1 has no exact binary representation, so a float sum can break or invent ties.
        reference = exact_posterior_reference(code, errors, 0.1)
        expected_ties = 0
        for error in errors:
            weights = brute_force_class_weights(code, error, 0.1)
            largest = max(weights)
            expected_ties += int(weights[0] == largest and weights.count(largest) > 1)
        self.assertEqual(reference.tie_shots, expected_ties)

    def test_uniform_noise_leaves_all_four_classes_exactly_tied(self) -> None:
        code = ToricCode(distance=2)
        errors = np.zeros((1, code.num_data_qubits), dtype=np.uint8)
        # At p=0.5 every chain is equally likely, so the four equal-sized classes tie exactly.
        reference = exact_posterior_reference(code, errors, 0.5)
        np.testing.assert_allclose(homology_class_posteriors(code, errors, 0.5), np.full((1, 4), 0.25))
        self.assertEqual(reference.tie_shots, 1)
        self.assertEqual(reference.exact_ml_p_fail[MaximumLikelihoodTieRule.TRIVIAL_WINS.value], 0.0)
        self.assertEqual(reference.exact_ml_p_fail[MaximumLikelihoodTieRule.TIES_FAIL.value], 1.0)
        self.assertAlmostEqual(reference.exact_ml_p_fail[MaximumLikelihoodTieRule.RANDOM.value], 0.75, places=12)

    def test_clean_syndromes_at_low_noise_never_fail_under_any_tie_rule(self) -> None:
        code = ToricCode(distance=4)
        errors = np.zeros((3, code.num_data_qubits), dtype=np.uint8)
        reference = exact_posterior_reference(code, errors, 0.01)
        self.assertEqual(reference.tie_shots, 0)
        for rule, rate in reference.exact_ml_p_fail.items():
            with self.subTest(rule=rule):
                self.assertEqual(rate, 0.0)
        self.assertLess(reference.ideal_sampling_p_fail, 1e-3)


@unittest.skipUnless(importlib.util.find_spec("pymatching") is not None, "optional PyMatching dependency is absent")
class MwpmTieSensitivityTest(unittest.TestCase):
    def test_agreement_rates_are_nested_and_account_for_every_shot(self) -> None:
        code = ToricCode(distance=4)
        rng = np.random.default_rng(5)
        errors = (rng.random((40, code.num_data_qubits)) < 0.10).astype(np.uint8)
        report = mwpm_tie_sensitivity(code, errors, code.syndrome(errors))
        self.assertEqual(report["compared_shots"] + report["skipped_above_exact_limit"], len(errors))
        self.assertGreater(report["compared_shots"], 0)
        # Identical recoveries imply the same homology, which implies the same failure outcome.
        self.assertLessEqual(report["identical_recovery_rate"], report["same_homology_rate"])
        self.assertLessEqual(report["same_homology_rate"], report["same_failure_outcome_rate"])
        self.assertLessEqual(report["same_failure_outcome_rate"], 1.0)
        self.assertAlmostEqual(
            report["p_fail_difference"], abs(report["exact_p_fail"] - report["pymatching_p_fail"]), places=12
        )

    def test_clean_syndromes_make_both_implementations_agree_exactly(self) -> None:
        code = ToricCode(distance=4)
        errors = np.zeros((5, code.num_data_qubits), dtype=np.uint8)
        report = mwpm_tie_sensitivity(code, errors, code.syndrome(errors))
        self.assertEqual(report["compared_shots"], 5)
        self.assertEqual(report["identical_recovery_rate"], 1.0)
        self.assertEqual(report["exact_p_fail"], 0.0)
        self.assertEqual(report["pymatching_p_fail"], 0.0)

    def test_shots_above_the_exact_defect_limit_leave_nothing_to_compare(self) -> None:
        code = ToricCode(distance=6)
        rng = np.random.default_rng(2)
        candidates = (rng.random((200, code.num_data_qubits)) < 0.5).astype(np.uint8)
        dense = candidates[code.syndrome(candidates).sum(axis=1) > 18]
        self.assertGreater(len(dense), 0, "expected at least one syndrome above the exact decoder's limit")
        with self.assertRaises(ValueError):
            mwpm_tie_sensitivity(code, dense, code.syndrome(dense))
