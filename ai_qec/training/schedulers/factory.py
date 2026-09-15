"""Scheduler factory placeholder."""

from __future__ import annotations


def scheduler_name(config: dict) -> str:
    """Read the scheduler name from training config."""
    return str(config.get("training", {}).get("scheduler", {}).get("name", "none"))
