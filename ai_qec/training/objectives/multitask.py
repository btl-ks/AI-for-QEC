"""Multitask objective bookkeeping."""

from __future__ import annotations


def weighted_sum(values: dict[str, float], weights: dict[str, float]) -> float:
    """Combine named objective terms with optional per-term weights."""
    return sum(values[key] * weights.get(key, 1.0) for key in values)
