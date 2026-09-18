#!/usr/bin/env python3
"""Redraw the logical-failure comparison against the paper's Fig. 3.

Everything plotted comes from ``results_summary.json``, so the figure stays in
step with the reference columns that ``torlai_melko_references.py`` writes there.
The L=4 panel shades the maximum-likelihood tie bracket, because the exact-ML
curve moves with the tie rule by more than most of the gaps under discussion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """Parse CLI arguments and redraw the two-panel comparison figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("paper/summary/assets/torlai_melko_2017/results_summary.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper/summary/assets/torlai_melko_2017/fig3_logical_failure.png"),
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    project_root = args.project_root.resolve()
    summary = json.loads((project_root / args.summary).read_text(encoding="utf-8"))

    def column(records: list[dict], key: str) -> np.ndarray:
        return np.array([record[key] for record in records], dtype=float)

    def error_bars(records: list[dict], key: str) -> np.ndarray:
        rate = column(records, key)
        low = np.array([record["ci"][0] for record in records], dtype=float)
        high = np.array([record["ci"][1] for record in records], dtype=float)
        return np.vstack((rate - low, high - rate))

    l4, l6_short, l6_long = summary["L4"], summary["L6_1000_steps"], summary["L6_10000_steps_redecode"]
    paper = summary["paper_read_off"]["fig3"]
    p4, p6 = column(l4, "p"), column(l6_short, "p")

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    tie_low = np.array([record["exact_ml_p_fail_tie_rules"]["trivial"] for record in l4])
    tie_high = np.array([record["exact_ml_p_fail_tie_rules"]["ties_fail"] for record in l4])
    left.fill_between(p4, tie_low, tie_high, color="0.6", alpha=0.22, linewidth=0,
                      label="Exact ML, tie-rule bracket")
    left.plot(p4, column(l4, "exact_ml_p_fail"), "--", color="black", linewidth=1.4,
              label="Exact maximum likelihood (random ties)")
    left.plot(p4, column(l4, "ideal_sampling_p_fail"), ":", color="#8c6bb1", linewidth=1.8,
              label="Ideal posterior sampling (enumerated)")
    left.plot(p4, column(l4, "mw"), "-", color="#e08214", linewidth=1.8, label="MWPM (ours)")
    left.plot(p4, np.array(paper["4"], dtype=float), "x", color="black", markersize=8,
              markeredgewidth=1.6, label="Paper Fig. 3 (read off, ±0.02)")
    left.errorbar(p4, column(l4, "rbm"), yerr=error_bars(l4, "rbm"), fmt="^", color="#c0392b",
                  markersize=7, capsize=3, linestyle="none", label="RBM, 400 steps (ours)")
    left.set_title("L = 4 (2000 held-out shots per point)")

    right.plot(p6, column(l6_short, "mw"), "-", color="#e08214", linewidth=1.8, label="MWPM (ours)")
    right.plot(p6, column(l6_short, "rbm"), "s", markerfacecolor="none", color="#4a9b5c",
               markersize=7, linestyle="none", label="RBM, 1000 steps (ours)")
    right.plot(p6, np.array(paper["6"], dtype=float), "x", color="black", markersize=8,
               markeredgewidth=1.6, label="Paper Fig. 3 (read off, ±0.02)")
    right.errorbar(p6, column(l6_long, "rbm"), yerr=error_bars(l6_long, "rbm"), fmt="s",
                   color="#2f7d4f", markersize=7, capsize=3, linestyle="none",
                   label="RBM, 10000 steps (ours)")
    right.set_title("L = 6 (1000 held-out shots per point)")

    for axis in (left, right):
        axis.set_xlabel("p_err")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8.5, loc="upper left")
    left.set_ylabel("logical failure probability P_fail")

    fig.tight_layout()
    out_path = project_root / args.out
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
