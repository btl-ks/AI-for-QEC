"""Multitask training for target regression and logical decoding."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from ai_qec.data.datasets.qec_dataset import load_split, validate_dataset
from ai_qec.models.decoders.transformer.model import LinearDetectorSummaryDecoder
from ai_qec.models.heads.multitask import multitask_metrics
from ai_qec.utils.config import config_hash, data_output_dir, write_json
from ai_qec.utils.reproducibility import code_version


def train_multitask(config: dict[str, Any], project_root: str | Path, run_dir: str | Path) -> dict[str, float]:
    """Train the numpy multitask baseline and save run checkpoints."""
    project_root = Path(project_root)
    run_path = Path(run_dir)
    dataset_dir = data_output_dir(config, project_root)
    dataset_manifest = validate_dataset(dataset_dir, config)

    train = load_split(dataset_dir, "train")
    validation = load_split(dataset_dir, "validation")

    alpha = float(config.get("training", {}).get("optimizer", {}).get("weight_decay", 1e-3))
    model = LinearDetectorSummaryDecoder(feature_names=train.feature_names).fit(
        train.features,
        train.target_strength,
        train.logical_label,
        ridge_alpha=alpha,
    )

    validation_pred = model.predict(validation.features)
    metrics = multitask_metrics(
        validation.target_strength,
        validation_pred["target_prediction"],
        validation.logical_label,
        validation_pred["logical_probability"],
        "validation",
    )

    checkpoint_dir = run_path / "checkpoints"
    model.save(
        checkpoint_dir / "best.npz",
        metadata={
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset_id": train.dataset_id,
            "trainer": "multitask_ridge",
            "selection": "single closed-form baseline; best and last are identical, not validation-selected",
            "model_implementation": config["model"]["implementation"],
            "feature_schema": train.feature_names,
            "dataset_config_hash": dataset_manifest["config_hash"],
            "dataset_generation_hash": dataset_manifest["generation_hash"],
            "code_version": code_version(project_root),
            "metrics": metrics,
        },
    )
    model.save(
        checkpoint_dir / "last.npz",
        metadata={
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "dataset_id": train.dataset_id,
            "trainer": "multitask_ridge",
            "selection": "single closed-form baseline; best and last are identical, not validation-selected",
            "model_implementation": config["model"]["implementation"],
            "feature_schema": train.feature_names,
            "dataset_config_hash": dataset_manifest["config_hash"],
            "dataset_generation_hash": dataset_manifest["generation_hash"],
            "code_version": code_version(project_root),
            "metrics": metrics,
        },
    )

    write_json(run_path / "training_summary.json", {"metrics": metrics, "dataset": str(dataset_dir), "checkpoint_selection": "single closed-form baseline"})
    return metrics
