#!/usr/bin/env python3
"""Run AI-QEC benchmarks for the current run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _bootstrap(project_root: Path) -> None:
    """Make the local package importable without installation."""
    sys.path.insert(0, str(project_root))


def _load_existing_metrics(path: Path) -> dict:
    """Load metrics.json if it exists so benchmarks can append to it."""
    # Return early when not path.exists().
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    # Choose the first expression when isinstance(data, dict); otherwise use the fallback.
    return data if isinstance(data, dict) else {}


def main() -> int:
    """Parse CLI arguments and run shortcut/robustness benchmarks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    _bootstrap(project_root)

    import numpy as np

    from ai_qec.benchmarks.decoding.toric import benchmark_toric_decoders
    from ai_qec.benchmarks.noise_learning.benchmark import shortcut_control_report
    from ai_qec.benchmarks.robustness.benchmark import nuisance_bucket_report
    from ai_qec.utils.config import first_seed, load_config, write_json

    config = load_config(args.config, project_root=project_root)
    # Follow this branch when config['data']['generator'] == 'toric_code_capacity'.
    if config["data"]["generator"] == "toric_code_capacity":
        report = benchmark_toric_decoders(config, args.run_dir)
        print(f"Benchmark report: {args.run_dir / 'benchmark_report.json'}")
        for key in sorted(report):
            print(f"{key}: {report[key]:.6g}")
        return 0
    pred_path = args.run_dir / "predictions" / "paper_eval.npz"
    # Reject this state when not pred_path.exists().
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    with np.load(pred_path, allow_pickle=False) as pred:
        target_true = pred["target_strength"]
        target_pred = pred["target_prediction"]
        depolarizing_rate = pred["depolarizing_rate"]
        measurement_rate = pred["measurement_rate"]
        nuisance_score = pred["nuisance_score"]

    report = {
        **shortcut_control_report(
            target_true=target_true,
            target_pred=target_pred,
            depolarizing_rate=depolarizing_rate,
            measurement_rate=measurement_rate,
            seed=first_seed(config),
        ),
        **nuisance_bucket_report(
            target_true=target_true,
            target_pred=target_pred,
            nuisance_score=nuisance_score,
        ),
    }

    metrics_path = args.run_dir / "metrics.json"
    metrics_doc = _load_existing_metrics(metrics_path)
    metrics_doc.setdefault("metrics", {}).update(report)
    write_json(metrics_path, metrics_doc)
    write_json(args.run_dir / "benchmark_report.json", {"benchmarks": report})

    print(f"Benchmark report: {args.run_dir / 'benchmark_report.json'}")
    for key in sorted(report):
        print(f"{key}: {report[key]:.6g}")
    return 0


# Run the command-line entry point when this module is executed directly.
if __name__ == "__main__":
    raise SystemExit(main())
