"""Cross-Attempt recovery contracts."""

from dataclasses import dataclass

from ai_qec.training.checkpoint.recovery import TrainingRecoveryCheckpoint

from .artifact import ArtifactRef


@dataclass(frozen=True, slots=True)
class RecoverySource:
    """Immutable identity of a terminal Attempt selected for recovery."""

    experiment_id: str
    source_attempt_id: str
    source_terminal_status: str


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    """Verified inputs allowed to seed a new Attempt."""

    source: RecoverySource
    completed_stage_artifacts: tuple[ArtifactRef, ...] = ()
    training_checkpoint: TrainingRecoveryCheckpoint | None = None
    resume_from_stage: str | None = None
