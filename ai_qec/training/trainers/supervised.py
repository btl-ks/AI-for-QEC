"""Supervised training alias."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_qec.training.trainers.multitask import train_multitask


def train_supervised(config: dict[str, Any], project_root: str | Path, run_dir: str | Path) -> dict[str, float]:
    """Train the current supervised baseline through the multitask trainer."""
    return train_multitask(config=config, project_root=project_root, run_dir=run_dir)
