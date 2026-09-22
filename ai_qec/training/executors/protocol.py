"""Dependency-free contract for one model-agnostic training update."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class TensorSignature:
    """Address-independent Tensor metadata used to select a captured graph."""

    field: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    stride: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BatchSignature:
    """Complete signature of the Tensor fields in one minibatch."""

    tensors: tuple[TensorSignature, ...]


@dataclass(frozen=True, slots=True)
class BatchSignatureEvidence:
    """Observed lifecycle and timings for one batch signature."""

    signature: BatchSignature
    warmup_steps: int
    capture_count: int
    replay_steps: int
    batch_copy_seconds: float
    capture_seconds: float
    replay_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingStepPlan:
    """Validated executor selection persisted in the resolved execution plan."""

    executor_id: str
    implementation_version: str
    device: str
    options: Mapping[str, object]
    options_digest: str


@dataclass(frozen=True, slots=True)
class TrainingStepContext:
    """Runtime objects required by either eager or captured training."""

    model: object
    visible: Callable[[Mapping[str, object]], object]
    objective: Callable[[object, object, object], object]
    optimizer: object
    generator: object
    device: str


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    """One successful minibatch update and its device-resident loss."""

    loss: object
    execution_mode: str


@dataclass(frozen=True, slots=True)
class TrainingStepEvidence:
    """Auditable evidence for the executor actually used by a Training Stage."""

    requested_executor: str
    observed_executor: str
    implementation_version: str
    device: str
    options_digest: str
    signatures: tuple[BatchSignatureEvidence, ...]
    warmup_steps: int
    capture_count: int
    replay_steps: int
    batch_copy_seconds: float
    capture_seconds: float
    replay_seconds: float
    fallback_observed: bool


@runtime_checkable
class TrainingStepExecutor(Protocol):
    """Perform exactly one configured optimizer update for each ``step`` call."""

    @property
    def plan(self) -> TrainingStepPlan: ...

    def step(self, tensors: Mapping[str, object]) -> TrainingStepResult: ...

    def finalize(self) -> None:
        """Reject an execution that cannot truthfully claim the selected mode."""

    def evidence(self) -> TrainingStepEvidence: ...
