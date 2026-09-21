"""Top-level experiment specification composed from domain contracts."""

from dataclasses import dataclass

from ai_qec.data.datasets.spec import DatasetSpec
from ai_qec.evaluation.scientific.spec import AccuracyGateSpec, ScientificEvaluationSpec
from ai_qec.models.spec import ModelSpec
from ai_qec.training.execution import ExecutionSpec
from ai_qec.training.spec import TrainingSpec


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Complete declarative inputs for one scientific experiment identity."""

    experiment_name: str
    dataset: DatasetSpec
    model: ModelSpec
    training: TrainingSpec
    execution: ExecutionSpec
    scientific_evaluation: ScientificEvaluationSpec
    accuracy_gate: AccuracyGateSpec
    master_seed: int
    schema_version: str = "experiment-spec-v1"
