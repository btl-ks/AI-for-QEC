"""Reuse a finished run's dataset and checkpoint in a new run that only decodes and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml

from ai_qec.data.datasets.toric_dataset import validate_toric_dataset
from ai_qec.utils.config import write_json

# Training settings a reusing run may change: they only affect decoding.
DECODE_ONLY_TRAINING_KEYS = ("decoder", "device")


@dataclass(frozen=True)
class SourceRun:
    """A successful run whose checkpoint and dataset are reused."""

    run_dir: Path
    config: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def recorded_sha256(self, path: Path) -> str:
        """Return the hash a stage recorded for ``path``; the last recording wins."""
        recorded = [
            artifact["sha256"]
            for step in self.manifest.get("steps", [])
            for artifact in step.get("artifacts", [])
            if Path(artifact["path"]) == path
        ]
        if not recorded:
            raise ValueError(f"Source run {self.run_id} did not record {path.name}")
        return recorded[-1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_run(run: str | Path, *, runs_root: str | Path) -> SourceRun:
    """Load a successful run by directory name (under ``runs_root``) or path."""
    run_dir = Path(run) if Path(run).is_dir() else Path(runs_root) / str(run)
    manifest_path, config_path = run_dir / "run_manifest.json", run_dir / "config.yaml"
    if not (manifest_path.is_file() and config_path.is_file()):
        raise FileNotFoundError(f"Not a run directory: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        raise ValueError(f"Source run {run_dir.name} has status {manifest.get('status')!r}, not 'success'")
    if not (run_dir / "checkpoints" / "best.pt").is_file():
        raise FileNotFoundError(f"Source run {run_dir.name} has no checkpoints/best.pt")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return SourceRun(run_dir.resolve(), config, manifest)


def check_reusable_config(source: SourceRun, config: dict[str, Any]) -> None:
    """Require the new config to match the source except for decode-only training settings."""
    differing = [key for key in ("qec", "noise", "data", "model") if config.get(key) != source.config.get(key)]
    new_training = {k: v for k, v in config["training"].items() if k not in DECODE_ONLY_TRAINING_KEYS}
    old_training = {k: v for k, v in source.config["training"].items() if k not in DECODE_ONLY_TRAINING_KEYS}
    if new_training != old_training:
        differing.append("training (other than decoder/device)")
    if differing:
        raise ValueError(f"Config differs from source run {source.run_id} in: {', '.join(differing)}")


def validate_source_dataset(source: SourceRun, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Validate the source dataset against the source config it was generated for.

    Dataset identity hashes the full run config, so a run that only changes decoding
    cannot validate the dataset against its own config.
    """
    check_reusable_config(source, config)
    dataset_dir = Path(source.manifest["dataset"]["dir"])
    return dataset_dir, validate_toric_dataset(dataset_dir, source.config)


def link_source_dataset(record: Any, source: SourceRun) -> None:
    """Point a run manifest at the source run's dataset, stating how it was validated."""
    record.manifest["dataset"] = {
        **source.manifest["dataset"],
        "config_hash_matches_run": False,
        "validated_against_source_run": source.run_id,
    }
    record.save()


def copy_source_checkpoint(
    source: SourceRun, destination: str | Path, *, config: dict[str, Any], dataset_manifest: dict[str, Any],
) -> Path:
    """Copy the source ``best.pt`` after checking its recorded hash and training dataset."""
    from ai_qec.models.registry import load_model

    checkpoint = source.run_dir / "checkpoints" / "best.pt"
    if _sha256(checkpoint) != source.recorded_sha256(checkpoint):
        raise RuntimeError(f"Checkpoint of source run {source.run_id} does not match its recorded hash")
    _, metadata = load_model(config, str(checkpoint))
    if (metadata.get("dataset_config_hash"), metadata.get("dataset_generation_hash")) != (
        dataset_manifest["config_hash"], dataset_manifest["generation_hash"],
    ):
        raise RuntimeError(f"Checkpoint of source run {source.run_id} was not trained on its dataset")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, target)
    return target


def source_training_summary(source: SourceRun) -> dict[str, Any]:
    """Return the source run's training summary, used for the training-curve plot."""
    path = source.run_dir / "training_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Source run {source.run_id} has no training_summary.json")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_with_source_predictions(
    source: SourceRun, *, config: dict[str, Any], syndromes: np.ndarray, recovery_valid: np.ndarray,
    gibbs_steps: np.ndarray, recovery: np.ndarray,
) -> dict[str, Any]:
    """Compare decoding with the source run over the step budget both runs share.

    With the same checkpoint, per-shot RNG, burn-in, chain count and device, the first
    ``min(max_steps)`` sweeps are identical, so every shot resolved within that budget
    must match exactly in both runs.
    """
    old, new = source.config["training"], config["training"]
    same_stream = (
        old.get("device", "cpu") == new.get("device", "cpu")
        and old["decoder"]["burn_in"] == new["decoder"]["burn_in"]
        and old["decoder"].get("parallel_chains", 1) == new["decoder"].get("parallel_chains", 1)
    )
    report: dict[str, Any] = {"source_run": source.run_id, "checked": same_stream}
    if not same_stream:
        report["reason"] = "device, burn_in or parallel_chains differ, so the random streams differ"
        return report
    with np.load(source.run_dir / "predictions" / "toric_rbm_eval.npz", allow_pickle=False) as payload:
        if not np.array_equal(payload["syndrome"], np.asarray(syndromes)):
            raise ValueError(f"Source run {source.run_id} predictions use a different test split")
        old_valid = payload["recovery_valid"].astype(bool)
        old_steps = payload["gibbs_steps"]
        old_recovery = payload["recovery"]
    new_valid = np.asarray(recovery_valid).astype(bool)
    new_steps, new_recovery = np.asarray(gibbs_steps), np.asarray(recovery)
    budget = min(int(old["decoder"]["max_steps"]), int(new["decoder"]["max_steps"]))
    old_within = old_valid & (old_steps <= budget)
    new_within = new_valid & (new_steps <= budget)
    identical = bool(
        np.array_equal(old_within, new_within)
        and np.array_equal(old_steps[old_within], new_steps[new_within])
        and np.array_equal(old_recovery[old_within], new_recovery[new_within])
    )
    report.update(
        shared_step_budget=budget,
        shots_resolved_within_budget=int(old_within.sum()),
        identical=identical,
        source_valid=int(old_valid.sum()),
        valid=int(new_valid.sum()),
    )
    return report


def verify_source_consistency(
    source: SourceRun, run_dir: str | Path, *, step: dict[str, Any], config: dict[str, Any],
    syndromes: np.ndarray, recovery_valid: np.ndarray, gibbs_steps: np.ndarray, recovery: np.ndarray,
) -> dict[str, Any]:
    """Compare decoding with the source run and fail the stage if it diverged.

    Same checkpoint and same per-shot RNG means the shared step budget must reproduce
    exactly, so any difference is a defect rather than sampling noise.
    """
    report = compare_with_source_predictions(
        source, config=config, syndromes=syndromes, recovery_valid=recovery_valid,
        gibbs_steps=gibbs_steps, recovery=recovery)
    path = Path(run_dir) / "source_consistency.json"
    write_json(path, report)
    step["outputs"].append(path)
    if report["checked"] and not report["identical"]:
        raise RuntimeError(f"decoding differs from the source run within the shared step budget: {report}")
    return report
