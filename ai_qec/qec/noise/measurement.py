"""Measurement-noise metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasurementNoise:
    """Metadata for measurement error probability."""
    probability: float
