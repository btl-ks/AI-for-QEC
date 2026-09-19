"""Contrastive-divergence training for the Torlai--Melko RBM decoder."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from ai_qec.data.datasets.toric_dataset import load_toric_split, validate_toric_dataset
from ai_qec.models.registry import build_model
from ai_qec.utils.config import write_json, data_output_dir, first_seed
from ai_qec.utils.reproducibility import code_version


BEST_CHECKPOINT_SELECTION = "lowest validation one-step reconstruction BCE"


@dataclass(frozen=True)
class RBMTrainingResult:
    """Training summary and files produced by the shared RBM training path."""

    summary: dict[str, Any]
    artifacts: tuple[Path, ...]


def rbm_checkpoint_metadata(
    config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    project_root: str | Path,
    *,
    metrics: dict[str, float],
    selection: str,
) -> dict[str, Any]:
    """Build checkpoint provenance shared by the best and final states."""
    return {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_id": dataset_manifest["dataset_id"],
        "trainer": "rbm_cd",
        "model_backend": "torch",
        "torch_version": str(torch.__version__),
        "visible_order": ["physical_error", "syndrome"],
        "selection": selection,
        "model_implementation": config["model"]["implementation"],
        "dataset_config_hash": dataset_manifest["config_hash"],
        "dataset_generation_hash": dataset_manifest["generation_hash"],
        "code_version": code_version(project_root),
        "metrics": metrics,
    }


def rbm_training_summary(
    history: list[dict[str, float]], *, selected_epoch: int | None, training_time_seconds: float, dataset_dir: str | Path,
) -> dict[str, Any]:
    """Summarize per-epoch reconstruction history and the selected checkpoint."""
    # Keep only values that satisfy the compound filter.
    selected = next((row for row in history if selected_epoch is not None and int(row["epoch"]) == int(selected_epoch)), None)
    # Reject this state when selected is None.
    if selected is None:
        raise ValueError("No epoch produced a finite validation reconstruction BCE; no best checkpoint exists")
    final = history[-1]
    return {
        "metrics": {
            "epochs": float(len(history)),
            "training_time_seconds": float(training_time_seconds),
            "best_validation_reconstruction_bce": float(selected["validation_reconstruction_bce"]),
            "final_train_reconstruction_bce": float(final["train_reconstruction_bce"]),
            "final_validation_reconstruction_bce": float(final["validation_reconstruction_bce"]),
            "selected_epoch": float(selected_epoch),
        },
        "history": history,
        "dataset": str(dataset_dir),
        "checkpoint_selection": BEST_CHECKPOINT_SELECTION,
    }


def run_rbm_training(config: dict[str, Any], project_root: str | Path, run_dir: str | Path) -> RBMTrainingResult:
    """Train from validated splits with one conversion and PyTorch minibatching."""
    root = Path(project_root)
    run_path = Path(run_dir)
    dataset_dir = data_output_dir(config, root)
    manifest = validate_toric_dataset(dataset_dir, config)
    train = load_toric_split(dataset_dir, "train")
    validation = load_toric_split(dataset_dir, "validation")
    training_cfg = config["training"]
    device = training_cfg.get("device", "cpu")
    model = build_model(
        config,
        error_units=train.physical_error.shape[1],
        syndrome_units=train.syndrome.shape[1],
    )
    model = model.to(device)
    epochs = int(training_cfg["epochs"])
    batch_size = int(training_cfg["batch_size"])
    cd_steps = int(training_cfg["cd_steps"])
    optimizer = torch.optim.SGD(model.parameters(), lr=float(training_cfg["learning_rate"]), weight_decay=float(training_cfg["weight_decay"]))
    train_visible = torch.as_tensor(train.visible, dtype=model.weights.dtype)
    validation_visible = model.to_visible_tensor(validation.visible)
    loader = DataLoader(
        TensorDataset(train_visible), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(first_seed(config) + 41),
        pin_memory=device == "cuda",
    )
    cd_rng = torch.Generator(device=device).manual_seed(first_seed(config) + 41)
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    best_epoch: int | None = None
    checkpoint_dir = run_path / "checkpoints"
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss_sum = torch.zeros((), dtype=model.weights.dtype, device=device)
        for (batch,) in loader:
            visible_batch = batch.to(device, non_blocking=device == "cuda")
            loss = model.contrastive_divergence_step(
                visible_batch, optimizer=optimizer, cd_steps=cd_steps,
                generator=cd_rng, validated=True, sync=False,
            )
            loss_sum += loss * len(batch)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_reconstruction_bce": float((loss_sum / len(train_visible)).item()),
            "validation_reconstruction_bce": model.reconstruction_bce(validation_visible),
        }
        history.append(epoch_metrics)
        # Follow this branch when epoch_metrics['validation_reconstruction_bce'] < best_validation.
        if epoch_metrics["validation_reconstruction_bce"] < best_validation:
            best_validation, best_epoch = epoch_metrics["validation_reconstruction_bce"], epoch
            save_rbm_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, config=config,
                dataset_manifest=manifest, project_root=root, metrics=epoch_metrics,
                selection=BEST_CHECKPOINT_SELECTION,
            )

    summary, outputs = finish_training(
        run_path, model, optimizer, config=config, dataset_manifest=manifest,
        project_root=root, history=history, best_epoch=best_epoch,
        training_time_seconds=time.perf_counter() - training_started, dataset_dir=dataset_dir,
    )
    return RBMTrainingResult(summary, (checkpoint_dir / "best.pt", *outputs))


def train_rbm(config: dict[str, Any], project_root: str | Path, run_dir: str | Path) -> dict[str, float]:
    """CLI-compatible entry point for the shared RBM training path."""
    return run_rbm_training(config, project_root, run_dir).summary["metrics"]


def save_rbm_checkpoint(
    path: str | Path, model: Any, optimizer: Any, *, config: dict[str, Any],
    dataset_manifest: dict[str, Any], project_root: str | Path, metrics: dict[str, Any], selection: str,
) -> Path:
    """Save one checkpoint together with the identity metadata that evaluation re-checks."""
    model.save(path, optimizer=optimizer, metadata=rbm_checkpoint_metadata(
        config, dataset_manifest, project_root, metrics=metrics, selection=selection))
    return Path(path)


def finish_training(
    run_dir: str | Path, model: Any, optimizer: Any, *, config: dict[str, Any],
    dataset_manifest: dict[str, Any], project_root: str | Path, history: list[dict[str, Any]],
    best_epoch: int, training_time_seconds: float, dataset_dir: str | Path,
) -> tuple[dict[str, Any], list[Path]]:
    """Write ``last.pt`` and the training summary, and return them for the stage record."""
    last = save_rbm_checkpoint(
        Path(run_dir) / "checkpoints" / "last.pt", model, optimizer, config=config,
        dataset_manifest=dataset_manifest, project_root=project_root,
        metrics=history[-1], selection="final epoch")
    summary = rbm_training_summary(
        history, selected_epoch=best_epoch, training_time_seconds=training_time_seconds,
        dataset_dir=dataset_dir)
    summary_path = Path(run_dir) / "training_summary.json"
    write_json(summary_path, summary)
    return summary, [last, summary_path]


def reuse_training_outputs(
    source_run: Any, *, run_dir: str | Path, config: dict[str, Any], dataset_manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    """Copy a source run's ``best.pt`` and training summary into this run.

    The checkpoint's recorded hash and its training dataset are both re-checked before
    the copy, so a reusing run cannot silently decode with a mismatched model.
    """
    from ai_qec.utils.source_run import copy_source_checkpoint, source_training_summary

    best = Path(run_dir) / "checkpoints" / "best.pt"
    copy_source_checkpoint(source_run, best, config=config, dataset_manifest=dataset_manifest)
    summary = source_training_summary(source_run)
    summary_path = Path(run_dir) / "training_summary.json"
    write_json(summary_path, summary)
    return summary, [best, summary_path]
