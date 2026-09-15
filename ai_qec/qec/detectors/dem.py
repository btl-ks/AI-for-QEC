"""Detector error model metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorErrorModel:
    """Minimal detector-error graph metadata."""
    num_detectors: int
    edges: list[tuple[int, int]]
