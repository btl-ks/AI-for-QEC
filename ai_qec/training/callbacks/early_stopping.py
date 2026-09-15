"""Early-stopping helper."""

from __future__ import annotations


def should_stop(wait: int, patience: int) -> bool:
    """Decide whether early stopping patience has been exhausted."""
    return wait >= patience
