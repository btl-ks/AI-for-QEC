#!/usr/bin/env python3
"""Train the configured AI-QEC model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _bootstrap(project_root: Path) -> None:
    """Make the local package importable without installation."""
    sys.path.insert(0, str(project_root))


def main() -> int:
    """Parse CLI arguments, train the model, and print validation metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    _bootstrap(project_root)

    from ai_qec.training.trainers.multitask import train_multitask
    from ai_qec.utils.config import load_config

    config = load_config(args.config, project_root=project_root)
    if config["training"]["trainer"] == "rbm_cd":
        from ai_qec.training.trainers.rbm import train_rbm
        metrics = train_rbm(config=config, project_root=project_root, run_dir=args.run_dir)
        checkpoint = args.run_dir / "checkpoints" / "best.pt"
    else:
        metrics = train_multitask(config=config, project_root=project_root, run_dir=args.run_dir)
        checkpoint = args.run_dir / "checkpoints" / "best.npz"
    print(f"Checkpoint: {checkpoint}")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
