"""Logical-failure comparison between the RBM and a toric MWPM reference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ai_qec.models.decoders.classical.mwpm import ExactToricMWPMDecoder
from ai_qec.models.decoders.protocol import DecodeRequest
from ai_qec.qec.codes.toric_code import ToricCode
from ai_qec.utils.config import write_json


def _wilson_interval(failures: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    """Return a 95% Wilson confidence interval for a Bernoulli rate."""
    if total < 1:
        raise ValueError("Wilson interval needs at least one sample")
    rate = failures / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return centre - half_width, centre + half_width


def _homology_counts(code: ToricCode, errors: np.ndarray, recoveries: np.ndarray) -> dict[str, int]:
    """Count the four closed-cycle homology sectors."""
    values = code.homology(errors ^ recoveries)
    return {
        "00": int(np.sum((values[:, 0] == 0) & (values[:, 1] == 0))),
        "01": int(np.sum((values[:, 0] == 0) & (values[:, 1] == 1))),
        "10": int(np.sum((values[:, 0] == 1) & (values[:, 1] == 0))),
        "11": int(np.sum((values[:, 0] == 1) & (values[:, 1] == 1))),
    }


def benchmark_toric_decoders(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """Measure logical failure of saved RBM outputs against exact small-L MWPM."""
    run_path = Path(run_dir)
    prediction_path = run_path / "predictions" / "toric_rbm_eval.npz"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as payload:
        required = {
            "split", "dataset_id", "lattice_size", "p_error", "parallel_chains", "device", "physical_error", "syndrome", "recovery",
            "recovery_valid", "timed_out", "gibbs_steps", "decoder_latency_ms", "logical_failure",
        }
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
    if parallel_chains != int(config["training"]["decoder"].get("parallel_chains", 1)) or device != config["training"].get("device", "cpu"):
        raise ValueError("Toric prediction decoder configuration mismatch")
    if not len(errors) or errors.shape != rbm_recoveries.shape or not np.array_equal(code.syndrome(errors), syndromes):
        raise ValueError("Toric predictions are inconsistent with the code")
    if decoder_latency_ms.shape != (len(errors),) or not np.isfinite(decoder_latency_ms).all() or np.any(decoder_latency_ms < 0):
        raise ValueError("Toric decoder latency data is invalid")
    if np.any(rbm_valid) and not np.array_equal(code.syndrome(rbm_recoveries[rbm_valid]), syndromes[rbm_valid]):
        raise ValueError("RBM prediction marked valid does not match its syndrome")
    mwpm = ExactToricMWPMDecoder(code)
    mwpm_recoveries = np.stack([mwpm.decode(DecodeRequest(syndrome, p_error)).recovery for syndrome in syndromes])
    mwpm_failures = code.logical_failure(errors, mwpm_recoveries).astype(bool)
    total = len(errors)
    rbm_low, rbm_high = _wilson_interval(int(np.sum(rbm_failures)), total)
    mwpm_low, mwpm_high = _wilson_interval(int(np.sum(mwpm_failures)), total)
    report: dict[str, Any] = {
        "split": split,
        "lattice_size": lattice_size,
        "p_error": p_error,
        "samples": total,
        "rbm": {
            "logical_failures": int(np.sum(rbm_failures)),
            "logical_error_rate": float(np.mean(rbm_failures)),
            "wilson_95": [rbm_low, rbm_high],
            "valid_recoveries": int(np.sum(rbm_valid)),
            "timeouts": int(total - np.sum(rbm_valid)),
            "parallel_chains": parallel_chains,
            "device": device,
            "decoder_latency_mean_ms": float(np.mean(decoder_latency_ms)),
            "decoder_latency_p50_ms": float(np.percentile(decoder_latency_ms, 50)),
            "decoder_latency_p95_ms": float(np.percentile(decoder_latency_ms, 95)),
            "homology_counts": _homology_counts(code, errors[rbm_valid], rbm_recoveries[rbm_valid]) if np.any(rbm_valid) else {"00": 0, "01": 0, "10": 0, "11": 0},
        },
        "mwpm_exact": {
            "logical_failures": int(np.sum(mwpm_failures)),
            "logical_error_rate": float(np.mean(mwpm_failures)),
            "wilson_95": [mwpm_low, mwpm_high],
            "homology_counts": _homology_counts(code, errors, mwpm_recoveries),
            "max_exact_defects": mwpm.max_exact_defects,
        },
    }
    metrics = {
        "rbm_logical_error_rate": report["rbm"]["logical_error_rate"],
        "rbm_timeout_rate": float(report["rbm"]["timeouts"] / total),
        "mwpm_logical_error_rate": report["mwpm_exact"]["logical_error_rate"],
    }
    metrics_path = run_path / "metrics.json"
    existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    if not isinstance(existing, dict):
        raise ValueError("metrics.json must contain an object")
    existing.setdefault("metrics", {}).update(metrics)
    write_json(metrics_path, existing)
    write_json(run_path / "benchmark_report.json", report)
    return metrics
