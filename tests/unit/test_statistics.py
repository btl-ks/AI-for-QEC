import unittest

import ai_qec.notebook_api as qec
from tests.helpers import requires_numpy

# Paired tables (both, candidate only, baseline only, neither) of the L=6 paper grid at
# p=0.05 and p=0.15, with the intervals the pre-scale implementation persisted for them.
L6_P005 = (247, 184, 137, 9432)
L6_P005_INTERVAL = (0.0011888718925160981, 0.008243495025758161)
L6_P015 = (4205, 1555, 1230, 3010)
L6_P015_INTERVAL = (0.022171794186487195, 0.04281544684075775)


class StatisticsTests(unittest.TestCase):
    def test_wilson_reference_values(self) -> None:
        low, high = qec.wilson_interval(10, 100, 0.95)
        self.assertAlmostEqual(low, 0.0552, places=4)
        self.assertAlmostEqual(high, 0.1744, places=4)
        self.assertEqual(qec.wilson_interval(0, 10, 0.95)[0], 0.0)
        self.assertAlmostEqual(qec.wilson_interval(0, 10, 0.95)[1], 0.2775, places=4)
        self.assertEqual(qec.wilson_interval(10, 10, 0.95)[1], 1.0)

    def test_wilson_rejects_invalid_counts(self) -> None:
        for successes, trials in ((1, 0), (-1, 5), (6, 5)):
            with self.subTest(successes=successes, trials=trials), self.assertRaises(ValueError):
                qec.wilson_interval(successes, trials, 0.95)
        with self.assertRaises(ValueError):
            qec.wilson_interval(1, 10, 1.0)

    def test_newcombe_paired_interval_properties(self) -> None:
        difference, low, high = qec.paired_difference_interval(12, 9, 2, 21, 0.95)
        self.assertAlmostEqual(difference, 7 / 44)
        self.assertLess(low, difference)
        self.assertGreater(high, difference)
        self.assertGreater(low, 0.0)
        mirrored = qec.paired_difference_interval(12, 2, 9, 21, 0.95)
        self.assertAlmostEqual(mirrored[0], -difference)
        self.assertAlmostEqual(mirrored[1], -high)
        self.assertAlmostEqual(mirrored[2], -low)

    def test_newcombe_narrows_with_more_shots_and_brackets_zero_without_discordance(self) -> None:
        _, small_low, small_high = qec.paired_difference_interval(10, 5, 5, 80, 0.95)
        _, large_low, large_high = qec.paired_difference_interval(100, 50, 50, 800, 0.95)
        self.assertLess(large_high - large_low, small_high - small_low)
        difference, low, high = qec.paired_difference_interval(30, 0, 0, 70, 0.95)
        self.assertEqual(difference, 0.0)
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_unit_scale_reproduces_persisted_newcombe_intervals_exactly(self) -> None:
        for table, interval in ((L6_P005, L6_P005_INTERVAL), (L6_P015, L6_P015_INTERVAL)):
            with self.subTest(table=table):
                _, low, high = qec.paired_difference_interval(*table, 0.95)
                self.assertEqual((low, high), interval)
                self.assertEqual(
                    qec.paired_difference_interval(*table, 0.95, scale=1.0),
                    qec.paired_difference_interval(*table, 0.95),
                )

    def test_scaled_contrast_follows_the_relative_margin(self) -> None:
        difference, low, high = qec.paired_difference_interval(*L6_P005, 0.95, scale=1.15)
        self.assertAlmostEqual(difference, (247 + 184) / 10000 - 1.15 * (247 + 137) / 10000)
        self.assertLess(low, difference)
        self.assertGreater(high, 0.0)
        _, _, high = qec.paired_difference_interval(*L6_P015, 0.95, scale=1.15)
        self.assertLess(high, 0.0)
        with self.assertRaises(ValueError):
            qec.paired_difference_interval(*L6_P005, 0.95, scale=0.0)

    @requires_numpy
    def test_pass_rate_on_the_relative_margin_matches_the_one_sided_level(self) -> None:
        import numpy as np

        rng = np.random.default_rng(2017)
        # Cell probabilities (both, first only, second only, neither) with p_first = 1.15 p_second.
        boundaries = {
            "low p": (0.030, 0.0275, 0.020, 0.9225),
            "high p": (0.450, 0.125, 0.050, 0.375),
        }
        for name, cells in boundaries.items():
            with self.subTest(boundary=name):
                draws = rng.multinomial(10_000, cells, size=4000)
                passes = sum(
                    qec.paired_difference_interval(*map(int, row), 0.95, scale=1.15)[2] <= 0.0
                    for row in draws
                )
                self.assertGreaterEqual(passes / len(draws), 0.015)
                self.assertLessEqual(passes / len(draws), 0.035)


if __name__ == "__main__":
    unittest.main()
