"""Pauli noise placeholders for future simulator backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PauliRates:
    """Independent Pauli error probabilities."""
    p_x: float = 0.0
    p_y: float = 0.0
    p_z: float = 0.0
