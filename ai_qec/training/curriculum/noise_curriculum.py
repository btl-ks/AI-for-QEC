"""Noise curriculum placeholders."""

from __future__ import annotations


def identity_schedule(values: list[float]) -> list[float]:
    """Return noise values unchanged for the initial curriculum baseline."""
    return values
