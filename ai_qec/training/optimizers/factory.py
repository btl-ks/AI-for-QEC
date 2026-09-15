"""Optimizer factory placeholder."""

from __future__ import annotations


def optimizer_name(config: dict) -> str:
    """Read the optimizer name from training config."""
    return str(config.get("training", {}).get("optimizer", {}).get("name", "ridge"))
