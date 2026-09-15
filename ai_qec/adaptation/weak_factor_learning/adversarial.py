"""Adversarial nuisance removal placeholder."""

from __future__ import annotations


def gradient_reversal_weight(enabled: bool, weight: float) -> float:
    """Return the adversarial nuisance-removal weight when enabled."""
    return float(weight if enabled else 0.0)
