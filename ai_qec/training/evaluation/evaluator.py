"""Evaluate checkpoints on generated splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_qec.data.datasets.qec_dataset import load_split, validate_dataset
from ai_qec.models.decoders.transformer.model import LinearDetectorSummaryDecoder
from ai_qec.models.heads.multitask import multitask_metrics
from ai_qec.utils.config import config_hash, data_output_dir, write_json
from ai_qec.utils.reproducibility import code_version


def evaluate_checkpoint(
    config: dict[str, Any],
    project_root: str | Path,
    run_dir: str | Path,
    checkpoint: str | Path,
    split: str = "test",
) -> dict[str, float]:
    """Evaluate a saved checkpoint and persist metrics plus predictions."""
    dataset_dir = data_output_dir(config, project_root)
    dataset_manifest = validate_dataset(dataset_dir, config)
    dataset = load_split(dataset_dir, split)
    model, metadata = LinearDetectorSummaryDecoder.load_with_metadata(checkpoint)
    expected = {
        "model_implementation": config["model"]["implementation"],
        "feature_schema": dataset.feature_names,
        "dataset_config_hash": dataset_manifest["config_hash"],
        "dataset_generation_hash": dataset_manifest["generation_hash"],
        "code_version": code_version(project_root),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint identity mismatch for {key}")
    pred = model.predict(dataset.features)

    metrics = multitask_metrics(
        dataset.target_strength,
        pred["target_prediction"],
        dataset.logical_label,
        pred["logical_probability"],
        split,
    )

    run_path = Path(run_dir)
    predictions_dir = run_path / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    with (predictions_dir / "paper_eval.npz").open("wb") as handle:
        np.savez(
            handle,
            split=np.asarray(split),
            dataset_id=np.asarray(dataset.dataset_id),
            feature_names=np.asarray(dataset.feature_names),
            features=dataset.features,
            target_strength=dataset.target_strength,
            target_prediction=pred["target_prediction"],
            logical_label=dataset.logical_label,
            logical_probability=pred["logical_probability"],
            depolarizing_rate=dataset.depolarizing_rate,
            measurement_rate=dataset.measurement_rate,
            nuisance_score=dataset.nuisance_score,
        )

    write_json(run_path / "metrics.json", {"metrics": metrics})
    return metrics
