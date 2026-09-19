"""Logical-failure comparison between the RBM and a toric MWPM reference."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ai_qec.models.decoders.classical.mwpm import ExactToricMWPMDecoder
from ai_qec.models.decoders.protocol import DecodeRequest
from ai_qec.qec.codes.stabilizer import StabilizerCode
from ai_qec.qec.codes.toric_code import ToricCode
from ai_qec.utils.config import write_json


def wilson_interval(failures: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    """Return a 95% Wilson confidence interval for a Bernoulli rate."""
    # Reject this state when total < 1.
    if total < 1:
        raise ValueError("Wilson interval needs at least one sample")
    rate = failures / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return centre - half_width, centre + half_width


def logical_class_labels(code: StabilizerCode) -> list[str]:
    """Return the ``2^k`` class labels, most significant logical bit first."""
    return ["".join(bits) for bits in itertools.product("01", repeat=code.num_logical_bits)]


def logical_class_counts(code: StabilizerCode, errors: np.ndarray, recoveries: np.ndarray) -> dict[str, int]:
    """Count how many residual cycles land in each logical class.

    For the toric code the two logical bits are the winding parities, so the labels
    are the familiar homology sectors ``00``-``11``; a code with one logical bit
    reports ``0`` and ``1`` instead.
    """
    labels = logical_class_labels(code)
    # Return early when not len(errors).
    if not len(errors):
        return dict.fromkeys(labels, 0)
    values = code.logical_class(errors ^ recoveries)
    weights = 1 << np.arange(code.num_logical_bits - 1, -1, -1)
    index = values.astype(np.int64) @ weights
    counts = np.bincount(index, minlength=len(labels))
    return {label: int(count) for label, count in zip(labels, counts, strict=True)}


def mwpm_reference_recoveries(code: ToricCode, syndromes: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return MWPM recoveries, the exact defect limit, and how many shots used the PyMatching fallback.

    Syndromes within the exact bitmask-DP limit are matched exactly; larger ones go to
    PyMatching, which also returns a minimum-weight perfect matching.
    """
    exact = ExactToricMWPMDecoder(code)
    fallback = None
    recoveries = []
    fallback_count = 0
    for syndrome in np.asarray(syndromes, dtype=np.uint8):
        # Follow this branch when int(syndrome.sum()) <= exact.max_exact_defects.
        if int(syndrome.sum()) <= exact.max_exact_defects:
            recoveries.append(exact.decode(syndrome))
            continue
        fallback_count += 1
        # Follow this branch when fallback is None.
        if fallback is None:
            from ai_qec.models.decoders.classical.pymatching_adapter import PyMatchingToricDecoder

            fallback = PyMatchingToricDecoder(code)
        recoveries.append(fallback.decode(DecodeRequest(syndrome)).recovery)
    return np.stack(recoveries), exact.max_exact_defects, fallback_count


def mwpm_tie_sensitivity(code: ToricCode, errors: np.ndarray, syndromes: np.ndarray) -> dict[str, Any]:
    """Measure how much the MWPM baseline depends on which minimum-weight matching is picked.

    Both decoders return a minimum-weight perfect matching, so they differ only when
    several matchings share that weight.  Degeneracy grows with the error rate, and a
    different choice can land in a different homology class, which moves ``P_fail``.
    Only shots within the exact decoder's defect limit can be compared.
    """
    from ai_qec.models.decoders.classical.pymatching_adapter import PyMatchingToricDecoder

    exact = ExactToricMWPMDecoder(code)
    matching = PyMatchingToricDecoder(code)
    errors = np.asarray(errors, dtype=np.uint8)
    compared = identical = same_homology = same_outcome = 0
    exact_failures = matching_failures = 0
    for error, syndrome in zip(errors, np.asarray(syndromes, dtype=np.uint8), strict=True):
        # Skip the current iteration when int(syndrome.sum()) > exact.max_exact_defects.
        if int(syndrome.sum()) > exact.max_exact_defects:
            continue
        exact_recovery = exact.decode(syndrome)
        matching_recovery = matching.decode(DecodeRequest(syndrome)).recovery
        # Reject this state when int(exact_recovery.sum()) != int(matching_recovery.sum()).
        if int(exact_recovery.sum()) != int(matching_recovery.sum()):
            raise RuntimeError("the two MWPM implementations disagree on the minimum weight")
        compared += 1
        identical += int(np.array_equal(exact_recovery, matching_recovery))
        same_homology += int(np.array_equal(
            code.homology(error ^ exact_recovery), code.homology(error ^ matching_recovery)
        ))
        exact_failed = bool(code.logical_failure(error[None, :], exact_recovery[None, :])[0])
        matching_failed = bool(code.logical_failure(error[None, :], matching_recovery[None, :])[0])
        exact_failures += int(exact_failed)
        matching_failures += int(matching_failed)
        same_outcome += int(exact_failed == matching_failed)
    # Reject this state when not compared.
    if not compared:
        raise ValueError("no shot was within the exact decoder's defect limit")
    return {
        "compared_shots": compared,
        "skipped_above_exact_limit": int(len(errors) - compared),
        "identical_recovery_rate": identical / compared,
        "same_homology_rate": same_homology / compared,
        "same_failure_outcome_rate": same_outcome / compared,
        "exact_p_fail": exact_failures / compared,
        "pymatching_p_fail": matching_failures / compared,
        "p_fail_difference": abs(exact_failures - matching_failures) / compared,
    }


def build_toric_benchmark_report(
    code: ToricCode, *, split: str, p_error: float, errors: np.ndarray, rbm_recoveries: np.ndarray,
    rbm_valid: np.ndarray, rbm_failures: np.ndarray, decoder_latency_ms: np.ndarray, parallel_chains: int,
    device: str, mwpm_recoveries: np.ndarray, max_exact_defects: int, mwpm_fallback_shots: int = 0,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Return the RBM-vs-MWPM report and its headline metrics for one decoded split."""
    rbm_valid = np.asarray(rbm_valid, dtype=bool)
    rbm_failures = np.asarray(rbm_failures, dtype=bool)
    mwpm_failures = code.logical_failure(errors, mwpm_recoveries).astype(bool)
    total = len(errors)
    report: dict[str, Any] = {
        "split": split,
        "lattice_size": code.distance,
        "p_error": float(p_error),
        "samples": total,
        "rbm": {
            "logical_failures": int(np.sum(rbm_failures)),
            "logical_error_rate": float(np.mean(rbm_failures)),
            "wilson_95": list(wilson_interval(int(np.sum(rbm_failures)), total)),
            "valid_recoveries": int(np.sum(rbm_valid)),
            "timeouts": int(total - np.sum(rbm_valid)),
            "parallel_chains": int(parallel_chains),
            "device": device,
            "decoder_latency_mean_ms": float(np.mean(decoder_latency_ms)),
            "decoder_latency_p50_ms": float(np.percentile(decoder_latency_ms, 50)),
            "decoder_latency_p95_ms": float(np.percentile(decoder_latency_ms, 95)),
            "homology_counts": logical_class_counts(code, errors[rbm_valid], rbm_recoveries[rbm_valid]),
        },
        "mwpm_exact": {
            "logical_failures": int(np.sum(mwpm_failures)),
            "logical_error_rate": float(np.mean(mwpm_failures)),
            "wilson_95": list(wilson_interval(int(np.sum(mwpm_failures)), total)),
            "homology_counts": logical_class_counts(code, errors, mwpm_recoveries),
            "max_exact_defects": int(max_exact_defects),
            "method": "exact bitmask-DP MWPM; PyMatching MWPM for syndromes above max_exact_defects",
            "pymatching_fallback_shots": int(mwpm_fallback_shots),
        },
    }
    metrics = {
        "rbm_logical_error_rate": report["rbm"]["logical_error_rate"],
        "rbm_timeout_rate": float(report["rbm"]["timeouts"] / total),
        "mwpm_logical_error_rate": report["mwpm_exact"]["logical_error_rate"],
    }
    return report, metrics


def write_benchmark_outputs(run_dir: str | Path, report: dict[str, Any], metrics: dict[str, float]) -> tuple[Path, Path]:
    """Merge benchmark metrics into ``metrics.json`` and write ``benchmark_report.json``."""
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.json"
    # Choose the first expression when metrics_path.is_file(); otherwise use the fallback.
    existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    # Reject this state when not isinstance(existing, dict).
    if not isinstance(existing, dict):
        raise ValueError("metrics.json must contain an object")
    existing.setdefault("metrics", {}).update(metrics)
    write_json(metrics_path, existing)
    report_path = run_path / "benchmark_report.json"
    write_json(report_path, report)
    return metrics_path, report_path


def benchmark_toric_decoders(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """Measure logical failure of saved RBM outputs against exact small-L MWPM."""
    run_path = Path(run_dir)
    prediction_path = run_path / "predictions" / "toric_rbm_eval.npz"
    # Reject this state when not prediction_path.is_file().
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as payload:
        required = {
            "split", "dataset_id", "lattice_size", "p_error", "parallel_chains", "device", "physical_error", "syndrome", "recovery",
            "recovery_valid", "timed_out", "gibbs_steps", "decoder_latency_ms", "logical_failure",
        }
        # Reject this state when set(payload.files) != required.
        if set(payload.files) != required:
            raise ValueError("Toric prediction schema mismatch")
        errors = payload["physical_error"].astype(np.uint8)
        syndromes = payload["syndrome"].astype(np.uint8)
        rbm_recoveries = payload["recovery"].astype(np.uint8)
        rbm_valid = payload["recovery_valid"].astype(bool)
        rbm_failures = payload["logical_failure"].astype(bool)
        decoder_latency_ms = payload["decoder_latency_ms"].astype(np.float64)
        lattice_size = int(payload["lattice_size"].item())
        p_error = float(payload["p_error"].item())
        parallel_chains = int(payload["parallel_chains"].item())
        device = str(payload["device"].item())
        split = str(payload["split"].item())

    code = ToricCode(distance=lattice_size)
    # Reject this state when the invalid compound condition is detected.
    if parallel_chains != int(config["training"]["decoder"].get("parallel_chains", 1)) or device != config["training"].get("device", "cpu"):
        raise ValueError("Toric prediction decoder configuration mismatch")
    # Reject this state when the invalid compound condition is detected.
    if not len(errors) or errors.shape != rbm_recoveries.shape or not np.array_equal(code.syndrome(errors), syndromes):
        raise ValueError("Toric predictions are inconsistent with the code")
    # Reject this state when the invalid compound condition is detected.
    if decoder_latency_ms.shape != (len(errors),) or not np.isfinite(decoder_latency_ms).all() or np.any(decoder_latency_ms < 0):
        raise ValueError("Toric decoder latency data is invalid")
    # Reject this state when the invalid compound condition is detected.
    if np.any(rbm_valid) and not np.array_equal(code.syndrome(rbm_recoveries[rbm_valid]), syndromes[rbm_valid]):
        raise ValueError("RBM prediction marked valid does not match its syndrome")
    mwpm_recoveries, max_exact_defects, mwpm_fallback_shots = mwpm_reference_recoveries(code, syndromes)
    report, metrics = build_toric_benchmark_report(
        code, split=split, p_error=p_error, errors=errors, rbm_recoveries=rbm_recoveries, rbm_valid=rbm_valid,
        rbm_failures=rbm_failures, decoder_latency_ms=decoder_latency_ms, parallel_chains=parallel_chains,
        device=device, mwpm_recoveries=mwpm_recoveries, max_exact_defects=max_exact_defects,
        mwpm_fallback_shots=mwpm_fallback_shots,
    )
    write_benchmark_outputs(run_path, report, metrics)
    return metrics
