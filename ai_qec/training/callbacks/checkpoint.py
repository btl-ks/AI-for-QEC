"""Checkpoint utilities."""

from __future__ import annotations

from pathlib import Path


def checkpoint_path(run_dir: str | Path, name: str = "best.npz") -> Path:
    """Return the expected checkpoint path inside a run directory."""
    return Path(run_dir) / "checkpoints" / name
