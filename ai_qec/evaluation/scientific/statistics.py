"""Binomial interval estimates used by scientific evaluation and the Accuracy Gate."""

from math import isfinite, sqrt
from statistics import NormalDist


def _z(confidence_level: float) -> float:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level!r}")
    return NormalDist().inv_cdf(0.5 + confidence_level / 2.0)


def wilson_interval(successes: int, trials: int, confidence_level: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError(f"invalid binomial counts {successes}/{trials}")
    z = _z(confidence_level)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, center - half_width)
    high = 1.0 if successes == trials else min(1.0, center + half_width)
    return low, high


def paired_difference_interval(
    both: int,
    first_only: int,
    second_only: int,
    neither: int,
    confidence_level: float,
    *,
    scale: float = 1.0,
) -> tuple[float, float, float]:
    """MOVER interval for ``p_first - scale * p_second`` on paired data.

    ``both``/``first_only``/``second_only``/``neither`` count shots on which both,
    only the first, only the second, or neither decoder failed. Each proportion
    uses its Wilson interval; ``scale * p_second`` has limits scaled by ``scale``
    and the same correlation with ``p_first`` (Zou & Donner 2008). ``scale = 1``
    is Newcombe (1998) method 10. Returns ``(difference, lower, upper)``.
    """

    counts = (both, first_only, second_only, neither)
    if any(count < 0 for count in counts) or sum(counts) == 0:
        raise ValueError(f"invalid paired counts {counts}")
    if not (scale > 0.0 and isfinite(scale)):
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    total = sum(counts)
    first = (both + first_only) / total
    second = (both + second_only) / total
    first_low, first_high = wilson_interval(both + first_only, total, confidence_level)
    second_low, second_high = wilson_interval(both + second_only, total, confidence_level)
    marginals = (
        (both + first_only)
        * (second_only + neither)
        * (both + second_only)
        * (first_only + neither)
    )
    phi = 0.0 if marginals == 0 else (both * neither - first_only * second_only) / sqrt(marginals)
    second_up = scale * (second_high - second)
    second_down = scale * (second - second_low)
    delta = sqrt(
        max(
            0.0,
            (first - first_low) ** 2
            - 2.0 * phi * (first - first_low) * second_up
            + second_up**2,
        )
    )
    epsilon = sqrt(
        max(
            0.0,
            (first_high - first) ** 2
            - 2.0 * phi * (first_high - first) * second_down
            + second_down**2,
        )
    )
    difference = first - scale * second
    return difference, max(-scale, difference - delta), min(1.0, difference + epsilon)
