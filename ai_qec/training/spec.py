"""Training semantics and trainer protocol."""

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from ai_qec.data.datasets.artifact import DatasetArtifact
from ai_qec.models.spec import ModelSpec

from .execution import ExecutionSpec


ModelT = TypeVar("ModelT")


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    """Learning semantics independent of hardware and process topology."""

    optimizer: str
    learning_rate: float
    epochs: int
    batch_size: int
    scheduler: str
    loss: str
    checkpoint_policy: str = "epoch-boundary"
    schema_version: str = "training-spec-v1"


@runtime_checkable
class Trainer(Protocol[ModelT]):
    """Framework adapter for training a model from an immutable dataset."""

    def train(
        self,
        model: ModelSpec,
        training: TrainingSpec,
        execution: ExecutionSpec,
        dataset: DatasetArtifact,
        attempt_id: str,
    ) -> ModelT: ...
