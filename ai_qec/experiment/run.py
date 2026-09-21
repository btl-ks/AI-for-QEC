"""Experiment and Attempt lifecycle protocols."""

from enum import StrEnum
from typing import Protocol, runtime_checkable

from .recovery import RecoveryPlan
from .spec import ExperimentSpec
from .stage import StageRecorder


class AttemptStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    PARTIAL = "partial"


@runtime_checkable
class Attempt(Protocol):
    """One immutable-history execution attempt within an Experiment."""

    @property
    def attempt_id(self) -> str: ...

    @property
    def status(self) -> AttemptStatus: ...

    @property
    def stages(self) -> StageRecorder: ...

    def complete(self) -> None: ...

    def fail(self, reason: str) -> None: ...

    def interrupt(self, reason: str) -> None: ...


@runtime_checkable
class Experiment(Protocol):
    """Creates new Attempts while retaining terminal execution history."""

    @property
    def experiment_id(self) -> str: ...

    @property
    def spec(self) -> ExperimentSpec: ...

    def start_attempt(self) -> Attempt: ...

    def recover(self, plan: RecoveryPlan) -> Attempt: ...


@runtime_checkable
class ExperimentFactory(Protocol):
    """Public construction boundary implemented by a future lifecycle runtime."""

    def create(self, spec: ExperimentSpec) -> Experiment: ...
