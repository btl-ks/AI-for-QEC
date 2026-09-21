"""Dependency-injected orchestration contracts for paper notebooks.

No lifecycle, dataset, training, evaluation, or rendering implementation lives
in this module. A future runtime must implement these Protocols.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from ai_qec.data.datasets.artifact import DatasetArtifact
from ai_qec.evaluation.scientific.result import (
    ScientificAcceptanceResult,
    ScientificEvaluationResult,
)
from ai_qec.experiment.artifact import ArtifactRef
from ai_qec.training.checkpoint.model import ModelCheckpoint


@runtime_checkable
class NotebookRun(Protocol):
    """High-level stage ordering boundary used by a paper notebook."""

    @property
    def attempt_id(self) -> str: ...

    def resolve_dataset(self) -> DatasetArtifact: ...

    def train(self, dataset: DatasetArtifact) -> ModelCheckpoint: ...

    def evaluate_accuracy(
        self,
        model: ModelCheckpoint,
        dataset: DatasetArtifact,
        baselines: Sequence[str],
    ) -> ScientificEvaluationResult: ...

    def check_accuracy_gate(
        self,
        result: ScientificEvaluationResult,
    ) -> ScientificAcceptanceResult: ...

    def evaluate_performance(
        self,
        model: ModelCheckpoint,
        dataset: DatasetArtifact,
    ) -> ArtifactRef: ...

    def visualize(
        self,
        scientific_result: ScientificEvaluationResult,
        acceptance: ScientificAcceptanceResult,
        performance_result: ArtifactRef | None,
    ) -> tuple[ArtifactRef, ...]: ...

    def finish(self) -> None: ...


@runtime_checkable
class NotebookExperiment(Protocol):
    """Start a new Attempt or create a recovery Attempt when appropriate."""

    def start_or_recover(self) -> NotebookRun: ...


@runtime_checkable
class NotebookPlatform(Protocol):
    """Public construction boundary injected into paper orchestration code."""

    def create_experiment(self, config: Mapping[str, object]) -> NotebookExperiment: ...
