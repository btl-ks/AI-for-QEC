"""Multi-objective score helpers."""

from __future__ import annotations


def weighted_objective(logical_error: float, qubits: float, runtime: float, decode_latency: float) -> float:
    """Compute a simple multi-objective design score."""
    return logical_error + 0.01 * qubits + 0.001 * runtime + 0.001 * decode_latency
