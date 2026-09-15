"""Contrastive-divergence training for the Torlai--Melko RBM decoder."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any
import time

import numpy as np
import torch

from ai_qec.data.datasets.toric_dataset import load_toric_split, validate_toric_dataset
from ai_qec.models.registry import build_model
from ai_qec.utils.config import data_output_dir, first_seed, write_json
from ai_qec.utils.reproducibility import code_version


def _metadata(
    *,
    config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    dataset_id: str,
    project_root: Path,
    metrics: dict[str, float],
    selection: str,
) -> dict[str, Any]:
    """Build checkpoint provenance shared by the best and final states."""
    return {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_id": dataset_id,
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


def train_rbm(config: dict[str, Any], project_root: str | Path, run_dir: str | Path) -> dict[str, float]:
    """Train a joint error--syndrome RBM with minibatch CD-k updates."""
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
    rng = np.random.default_rng(first_seed(config) + 41)
    torch_rng = torch.Generator(device=device).manual_seed(first_seed(config) + 41)
    visible = train.visible
    validation_visible = validation.visible
    epochs = int(training_cfg["epochs"])
    batch_size = int(training_cfg["batch_size"])
    learning_rate = float(training_cfg["learning_rate"])
    cd_steps = int(training_cfg["cd_steps"])
    weight_decay = float(training_cfg["weight_decay"])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    checkpoint_dir = run_path / "checkpoints"
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(visible))
        batch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_losses.append(
                model.contrastive_divergence_step(
                    visible[indices],
                    optimizer=optimizer,
                    cd_steps=cd_steps,
                    generator=torch_rng,
                )
            )
        train_loss = float(np.mean(batch_losses))
        validation_loss = model.reconstruction_bce(validation_visible)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_reconstruction_bce": train_loss,
            "validation_reconstruction_bce": validation_loss,
        }
        history.append(epoch_metrics)
        if validation_loss < best_validation:
            best_validation = validation_loss
            model.save(
                checkpoint_dir / "best.pt",
                optimizer=optimizer,
                metadata=_metadata(
                    config=config,
                    dataset_manifest=manifest,
                    dataset_id=train.dataset_id,
                    project_root=root,
                    metrics=epoch_metrics,
                    selection="lowest validation one-step reconstruction BCE",
                ),
            )

    final_metrics = dict(history[-1])
    model.save(
        checkpoint_dir / "last.pt",
        optimizer=optimizer,
        metadata=_metadata(
            config=config,
            dataset_manifest=manifest,
            dataset_id=train.dataset_id,
            project_root=root,
            metrics=final_metrics,
            selection="final epoch",
        ),
    )
    best_model, best_metadata = type(model).load_with_metadata(checkpoint_dir / "best.pt")
    best_metrics = {
        "epochs": float(epochs),
        "training_time_seconds": float(time.perf_counter() - training_started),
        "best_validation_reconstruction_bce": best_model.reconstruction_bce(validation_visible),
        "final_train_reconstruction_bce": final_metrics["train_reconstruction_bce"],
        "final_validation_reconstruction_bce": final_metrics["validation_reconstruction_bce"],
        "selected_epoch": float(best_metadata["metrics"]["epoch"]),
    }
    write_json(
        run_path / "training_summary.json",
        {
            "metrics": best_metrics,
            "history": history,
            "dataset": str(dataset_dir),
            "checkpoint_selection": "lowest validation one-step reconstruction BCE",
        },
    )
    return best_metrics
