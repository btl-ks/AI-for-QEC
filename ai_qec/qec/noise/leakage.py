"""Leakage-noise metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeakageNoise:
    """Metadata for leakage probability."""
    probability: float = 0.0
