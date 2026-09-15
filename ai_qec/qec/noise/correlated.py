"""Correlated-noise metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorrelatedNoise:
    """Metadata for spatial and temporal correlation strengths."""
    spatial_correlation: float = 0.0
    temporal_correlation: float = 0.0
