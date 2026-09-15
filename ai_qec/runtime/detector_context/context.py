"""Detector context."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorContext:
    """Runtime metadata for detector count and future connectivity."""
    num_detectors: int
