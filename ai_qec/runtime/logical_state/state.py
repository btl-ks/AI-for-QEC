"""Logical state metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LogicalState:
    """Track the number of logical qubits in a runtime state."""
    logical_qubits: int
