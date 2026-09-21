"""Stage state and recording protocols."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .artifact import ArtifactRef


class StageStatus(StrEnum):
    NOT_STARTED = "not-started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class StageRecord:
    """Persisted state at a recoverable experiment boundary."""

    stage_id: str
    name: str
    status: StageStatus
    started_at: str | None = None
    finished_at: str | None = None
    input_artifacts: tuple[ArtifactRef, ...] = ()
    output_artifacts: tuple[ArtifactRef, ...] = ()
    error: str | None = None
    recovery_metadata_uri: str | None = None


@runtime_checkable
class StageRecorder(Protocol):
    """Persistence boundary for StageRecord transitions."""

    def get(self, attempt_id: str, stage_id: str) -> StageRecord | None: ...

    def record(self, attempt_id: str, stage: StageRecord) -> None: ...
