"""Logging helpers."""

from __future__ import annotations


def format_metric(name: str, value: float) -> str:
    """Format one metric for concise CLI logging."""
    return f"{name}={value:.6g}"
