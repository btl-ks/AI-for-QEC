"""Throughput profiling."""

from __future__ import annotations


def examples_per_second(n_examples: int, seconds: float) -> float:
    """Compute throughput while guarding against division by zero."""
    return float(n_examples / max(seconds, 1e-12))
