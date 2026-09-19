#!/usr/bin/env python3
"""Recompute the exported scope: test target-regression metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/best.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/paper_eval.npz"))
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from model import LinearDetectorSummaryDecoder
    from regression_head import regression_metrics

    model = LinearDetectorSummaryDecoder.load(args.checkpoint)
    with np.load(args.data, allow_pickle=False) as data:
        recomputed = regression_metrics(data["target_strength"], model.predict_target(data["features"]), "test")
    saved = json.loads(args.metrics.read_text(encoding="utf-8"))["metrics"]
    # Keep only values that satisfy the compound filter.
    mismatches = {key: (value, saved.get(key)) for key, value in recomputed.items() if not isinstance(saved.get(key), (int, float)) or abs(value - saved[key]) > args.atol}
    # Reject this state when mismatches.
    if mismatches:
        raise SystemExit(f"Metric verification failed: {mismatches}")
    print(json.dumps(recomputed, indent=2, sort_keys=True))
    return 0


# Run the command-line entry point when this module is executed directly.
if __name__ == "__main__":
    raise SystemExit(main())
