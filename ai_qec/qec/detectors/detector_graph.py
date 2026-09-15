"""Detector graph helpers."""

from __future__ import annotations


def line_adjacency(num_detectors: int) -> list[tuple[int, int]]:
    """Create nearest-neighbor detector edges on a 1D line."""
    return [(i, i + 1) for i in range(max(0, num_detectors - 1))]
