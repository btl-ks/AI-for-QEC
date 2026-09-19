#!/usr/bin/env python3
"""Recompute the model-free reference columns of the Torlai-Melko summary.

Three quantities in ``paper/summary`` are not produced by any single run, because
they depend only on the code and the test errors rather than on a trained model:

* the exact maximum-likelihood failure rate at L=4, under all three tie rules,
* the ideal-posterior-sampling failure rate at L=4,
* how far the L=6 MWPM baseline depends on which minimum-weight matching is picked.

This script reads the saved per-shot predictions of the runs listed in
``results_summary.json`` and writes the results back into that file, so the
summary's reference columns can be regenerated from the committed artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _bootstrap(project_root: Path) -> None:
    """Make the local package importable without installation."""
    sys.path.insert(0, str(project_root))


def main() -> int:
    """Parse CLI arguments and refresh the summary's reference columns."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("paper/summary/assets/torlai_melko_2017/results_summary.json"),
    )
    parser.add_argument("--skip-mwpm-ties", action="store_true", help="skip the slow L=6 comparison")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    _bootstrap(project_root)

    import numpy as np

    from ai_qec.benchmarks.decoding.exact_posterior import exact_posterior_reference
    from ai_qec.benchmarks.decoding.toric import mwpm_tie_sensitivity
    from ai_qec.qec.codes.toric_code import ToricCode

    summary_path = (project_root / args.summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs_root = project_root / "runs"

    def predictions(run_id: str) -> dict[str, np.ndarray]:
        path = runs_root / run_id / "predictions" / "toric_rbm_eval.npz"
        # Reject this state when not path.is_file().
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            return {"physical_error": payload["physical_error"], "syndrome": payload["syndrome"]}

    code_l4 = ToricCode(distance=4)
    for record in summary["L4"]:
        data = predictions(record["run"])
        reference = exact_posterior_reference(code_l4, data["physical_error"], record["p"])
        record["exact_ml_p_fail"] = reference.exact_ml_p_fail["random"]
        record["exact_ml_p_fail_tie_rules"] = reference.exact_ml_p_fail
        record["exact_ml_tie_shots"] = reference.tie_shots
        record["ideal_sampling_p_fail"] = reference.ideal_sampling_p_fail
        print(
            f"L=4 p={record['p']:.2f} exact_ml={record['exact_ml_p_fail']:.4f} "
            f"(bracket {reference.exact_ml_p_fail['trivial']:.4f}-{reference.exact_ml_p_fail['ties_fail']:.4f}, "
            f"{reference.tie_shots} ties) ideal={reference.ideal_sampling_p_fail:.4f}",
            flush=True,
        )

    # Follow this branch when not args.skip_mwpm_ties.
    if not args.skip_mwpm_ties:
        code_l6 = ToricCode(distance=6)
        summary["mwpm_tie_sensitivity_L6"] = []
        for record in summary["L6_1000_steps"]:
            data = predictions(record["run"])
            result = mwpm_tie_sensitivity(code_l6, data["physical_error"], data["syndrome"])
            summary["mwpm_tie_sensitivity_L6"].append({"p": record["p"], **result})
            print(
                f"L=6 p={record['p']:.2f} same_homology={result['same_homology_rate']:.3f} "
                f"same_outcome={result['same_failure_outcome_rate']:.3f} "
                f"dP_fail={result['p_fail_difference']:.4f}",
                flush=True,
            )

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {summary_path}")
    return 0


# Run the command-line entry point when this module is executed directly.
if __name__ == "__main__":
    raise SystemExit(main())
