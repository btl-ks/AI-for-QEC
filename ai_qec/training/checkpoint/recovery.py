"""Checkpoint used exclusively for creating a recovery Attempt."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingRecoveryCheckpoint:
    """Reference to complete epoch-boundary training recovery state."""

    checkpoint_id: str
    artifact_id: str
    source_attempt_id: str
    epoch: int
    global_step: int
    uri: str
    checksum: str
    includes_optimizer: bool
    includes_scheduler: bool
    includes_amp_scaler: bool
    includes_rng_state: bool
    includes_data_cursor: bool
    schema_version: str = "training-recovery-checkpoint-v1"
