"""Memory profiling placeholder."""

from __future__ import annotations


def bytes_to_mebibytes(value: int) -> float:
    """Convert a byte count to mebibytes."""
    return float(value) / (1024.0 * 1024.0)
