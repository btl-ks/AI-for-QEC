"""Evaluation of a syndrome-clamped RBM decoder on raw toric test data."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import numpy as np

from ai_qec.data.datasets.toric_dataset import ToricDataset, load_toric_split, validate_toric_dataset
from ai_qec.models.decoders.generative.rbm_decoder import RBMGibbsDecoder
from ai_qec.models.decoders.protocol import DecodeRequest
from ai_qec.models.registry import load_model
from ai_qec.qec.codes.toric_code import ToricCode
from ai_qec.utils.config import data_output_dir, first_seed, write_json
from ai_qec.utils.reproducibility import code_version


def decoding_rng(config: dict[str, Any], index: int) -> np.random.Generator:
    """Return the reproducible, independent RNG stream for decoding sample ``index``."""
    return np.random.default_rng(first_seed(config) + 1_000_000 + index)


def rbm_decoding_metrics(
    split: str, *, logical_failure: np.ndarray, timed_out: np.ndarray, recovery_valid: np.ndarray,
    gibbs_steps: np.ndarray, decoder_latency_ms: np.ndarray,
) -> dict[str, float]:
    """Summarize per-sample RBM decoding outcomes; timeouts are already counted as failures."""
    return {
        f"{split}_rbm_logical_error_rate": float(np.mean(logical_failure)),
        f"{split}_rbm_timeout_rate": float(np.mean(timed_out)),
        f"{split}_rbm_valid_recovery_rate": float(np.mean(recovery_valid)),
        f"{split}_rbm_mean_gibbs_steps": float(np.mean(gibbs_steps)),
        f"{split}_rbm_p95_gibbs_steps": float(np.percentile(gibbs_steps, 95)),
        f"{split}_rbm_decoder_latency_mean_ms": float(np.mean(decoder_latency_ms)),
        f"{split}_rbm_decoder_latency_p50_ms": float(np.percentile(decoder_latency_ms, 50)),
        f"{split}_rbm_decoder_latency_p95_ms": float(np.percentile(decoder_latency_ms, 95)),
    }


class DecodingRecord:
    """Collect per-shot decoding outcomes for one split, then emit metrics and predictions.

    Every shot starts marked as a logical failure, so a timeout counts as one without
    the caller having to remember: only a recovery that is found and checked can clear
    the flag.  Callers keep their own per-shot loop and report each outcome here.
    """

    def __init__(self, n_samples: int, num_data_qubits: int) -> None:
        # Reject this state when n_samples < 1 or num_data_qubits < 1.
        if n_samples < 1 or num_data_qubits < 1:
            raise ValueError("n_samples and num_data_qubits must be positive")
        self.recovery = np.zeros((n_samples, num_data_qubits), dtype=np.uint8)
        self.recovery_valid = np.zeros(n_samples, dtype=np.uint8)
        self.timed_out = np.zeros(n_samples, dtype=np.uint8)
        self.gibbs_steps = np.zeros(n_samples, dtype=np.int32)
        self.decoder_latency_ms = np.zeros(n_samples, dtype=np.float64)
        self.logical_failure = np.ones(n_samples, dtype=np.uint8)

    def add(
        self, index: int, *, recovery: np.ndarray | None, steps: int, latency_ms: float,
        failed: bool | None = None,
    ) -> None:
        """Record one shot; ``recovery=None`` is a timeout and stays a logical failure."""
        self.gibbs_steps[index] = steps
        self.decoder_latency_ms[index] = latency_ms
        # Follow this branch when recovery is None.
        if recovery is None:
            self.timed_out[index] = 1
            return
        # Reject this state when failed is None.
        if failed is None:
            raise ValueError("a decoded shot needs its logical-failure outcome")
        self.recovery[index] = recovery
        self.recovery_valid[index] = 1
        self.logical_failure[index] = np.uint8(failed)

    def metrics(self, split: str) -> dict[str, float]:
        """Return the headline decoding metrics for this split."""
        return rbm_decoding_metrics(
            split, logical_failure=self.logical_failure, timed_out=self.timed_out,
            recovery_valid=self.recovery_valid, gibbs_steps=self.gibbs_steps,
            decoder_latency_ms=self.decoder_latency_ms,
        )

    def save(self, path: str | Path, *, dataset: ToricDataset, parallel_chains: int, device: str) -> None:
        """Persist the per-shot predictions in the schema the benchmark reads."""
        save_toric_predictions(
            path, dataset=dataset, parallel_chains=parallel_chains, device=device,
            recovery=self.recovery, recovery_valid=self.recovery_valid, timed_out=self.timed_out,
            gibbs_steps=self.gibbs_steps, decoder_latency_ms=self.decoder_latency_ms,
            logical_failure=self.logical_failure,
        )


def save_toric_predictions(
    path: str | Path, *, dataset: ToricDataset, parallel_chains: int, device: str, recovery: np.ndarray,
    recovery_valid: np.ndarray, timed_out: np.ndarray, gibbs_steps: np.ndarray, decoder_latency_ms: np.ndarray,
    logical_failure: np.ndarray,
) -> None:
    """Persist per-sample predictions in the schema read by the toric benchmark."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        np.savez_compressed(
            handle,
            split=np.asarray(dataset.split),
            dataset_id=np.asarray(dataset.dataset_id),
            lattice_size=np.asarray(dataset.lattice_size),
            p_error=np.asarray(dataset.p_error),
            parallel_chains=np.asarray(parallel_chains),
            device=np.asarray(device),
            physical_error=dataset.physical_error,
            syndrome=dataset.syndrome,
            recovery=recovery,
            recovery_valid=recovery_valid,
            timed_out=timed_out,
            gibbs_steps=gibbs_steps,
            decoder_latency_ms=decoder_latency_ms,
            logical_failure=logical_failure,
        )


def evaluate_toric_rbm(
    config: dict[str, Any],
    project_root: str | Path,
    run_dir: str | Path,
    checkpoint: str | Path,
    split: str = "test",
) -> dict[str, float]:
    """Decode every requested split sample and persist traceable predictions."""
    root = Path(project_root)
    dataset_dir = data_output_dir(config, root)
    manifest = validate_toric_dataset(dataset_dir, config)
    dataset = load_toric_split(dataset_dir, split)
    model, metadata = load_model(config, str(checkpoint))
    expected = {
        "model_implementation": config["model"]["implementation"],
        "model_backend": "torch",
        "visible_order": ["physical_error", "syndrome"],
        "dataset_config_hash": manifest["config_hash"],
        "dataset_generation_hash": manifest["generation_hash"],
        "code_version": code_version(root),
    }
    for key, value in expected.items():
        # Reject this state when metadata.get(key) != value.
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint identity mismatch for {key}")

    code = ToricCode(distance=dataset.lattice_size)
    decoder_cfg = config["training"]["decoder"]
    decoder = RBMGibbsDecoder(
        model,
        code,
        burn_in=int(decoder_cfg["burn_in"]),
        max_steps=int(decoder_cfg["max_steps"]),
        parallel_chains=int(decoder_cfg.get("parallel_chains", 1)),
        device=config["training"].get("device", "cpu"),
    )
    recoveries = np.zeros_like(dataset.physical_error, dtype=np.uint8)
    recovery_valid = np.zeros(len(dataset.physical_error), dtype=np.uint8)
    timed_out = np.zeros(len(dataset.physical_error), dtype=np.uint8)
    gibbs_steps = np.zeros(len(dataset.physical_error), dtype=np.int32)
    decoder_latency_ms = np.zeros(len(dataset.physical_error), dtype=np.float64)
    logical_failure = np.ones(len(dataset.physical_error), dtype=np.uint8)
    for index, (error, syndrome) in enumerate(zip(dataset.physical_error, dataset.syndrome, strict=True)):
        started = time.perf_counter_ns()
        result = decoder.decode(DecodeRequest(syndrome, dataset.p_error), rng=decoding_rng(config, index))
        decoder_latency_ms[index] = (time.perf_counter_ns() - started) / 1e6
        gibbs_steps[index] = result.steps
        # Follow this branch when result.recovery is None.
        if result.recovery is None:
            timed_out[index] = 1
            continue
        recoveries[index] = result.recovery
        recovery_valid[index] = 1
        logical_failure[index] = np.uint8(code.logical_failure(error[None, :], result.recovery[None, :])[0])

    metrics = rbm_decoding_metrics(
        split, logical_failure=logical_failure, timed_out=timed_out, recovery_valid=recovery_valid,
        gibbs_steps=gibbs_steps, decoder_latency_ms=decoder_latency_ms,
    )
    run_path = Path(run_dir)
    save_toric_predictions(
        run_path / "predictions" / "toric_rbm_eval.npz", dataset=dataset, parallel_chains=decoder.parallel_chains,
        device=decoder.device, recovery=recoveries, recovery_valid=recovery_valid, timed_out=timed_out,
        gibbs_steps=gibbs_steps, decoder_latency_ms=decoder_latency_ms, logical_failure=logical_failure,
    )
    write_json(run_path / "metrics.json", {"metrics": metrics})
    return metrics
