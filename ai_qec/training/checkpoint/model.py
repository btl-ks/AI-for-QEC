"""Checkpoint used for inference, evaluation, reuse, or deployment."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelCheckpoint:
    """Portable model state reference; never implies resumable training state."""

    checkpoint_id: str
    artifact_id: str
    model_identity: str
    dataset_artifact_id: str
    uri: str
    checksum: str
    selected_metric: str | None = None
    schema_version: str = "model-checkpoint-v1"
