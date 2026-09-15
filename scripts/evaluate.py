#!/usr/bin/env python3
"""Evaluate a trained AI-QEC checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _bootstrap(project_root: Path) -> None:
    """Make the local package importable without installation."""
    sys.path.insert(0, str(project_root))


def main() -> int:
    """Parse CLI arguments and evaluate a checkpoint on a dataset split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    _bootstrap(project_root)

    from ai_qec.training.evaluation.evaluator import evaluate_checkpoint
    from ai_qec.utils.config import load_config

    config = load_config(args.config, project_root=project_root)
    if config["training"]["trainer"] == "rbm_cd":
        from ai_qec.training.evaluation.toric_rbm import evaluate_toric_rbm
        metrics = evaluate_toric_rbm(
            config=config,
            project_root=project_root,
            run_dir=args.run_dir,
            checkpoint=args.checkpoint,
            split=args.split,
        )
        prediction_path = args.run_dir / "predictions" / "toric_rbm_eval.npz"
    else:
        metrics = evaluate_checkpoint(
            config=config,
            project_root=project_root,
            run_dir=args.run_dir,
            checkpoint=args.checkpoint,
            split=args.split,
        )
        prediction_path = args.run_dir / "predictions" / "paper_eval.npz"
    print(f"Predictions: {prediction_path}")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
