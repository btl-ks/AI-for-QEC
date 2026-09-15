"""Compiled circuit context."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompiledCircuitContext:
    """Runtime metadata for a compiled QEC circuit."""
    name: str
    rounds: int
