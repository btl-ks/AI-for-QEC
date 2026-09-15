"""Adversarial nuisance objective placeholders."""

from __future__ import annotations


def nuisance_penalty_enabled(weight: float) -> bool:
    """Return whether an adversarial nuisance penalty should be active."""
    return weight > 0.0
