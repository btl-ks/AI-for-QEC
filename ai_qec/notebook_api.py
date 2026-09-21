"""Stable public facade for notebooks and external experiment drivers.

Only contracts are exported in the bootstrap repository. Concrete factories,
generators, trainers, decoders, evaluators, and gates arrive through future
OpenSpec-governed implementations.
"""

from ai_qec.data.datasets import (
    DatasetArtifact,
    DatasetIdentityProvider,
    DatasetInstance,
    DatasetKey,
    DatasetRegistry,
    DatasetResolution,
    DatasetResolver,
    DatasetRole,
    DatasetSpec,
    DatasetSplit,
)
from ai_qec.data.generators import QECDataGenerator
from ai_qec.data.schema import QECBatch
from ai_qec.evaluation.scientific import (
    AccuracyGate,
    AccuracyGateSpec,
    DecoderEvaluation,
    GateDecision,
    MetricEstimate,
    ScientificAcceptanceResult,
    ScientificEvaluationResult,
    ScientificEvaluationSpec,
    ScientificEvaluator,
)
from ai_qec.experiment import (
    ArtifactKind,
    ArtifactManifest,
    ArtifactRef,
    ArtifactRepository,
    Attempt,
    AttemptStatus,
    Experiment,
    ExperimentFactory,
    ExperimentSpec,
    RandomStreamDescriptor,
    RandomStreams,
    RecoveryPlan,
    RecoverySource,
    StageRecord,
    StageRecorder,
    StageStatus,
)
from ai_qec.models import ModelSpec
from ai_qec.models.decoders import DecodeRequest, DecodeResult, DecodeStatus, Decoder
from ai_qec.paper import NotebookExperiment, NotebookPlatform, NotebookRun
from ai_qec.qec import NoiseApproximation, NoiseSpec, QECSpec
from ai_qec.qec.backends import BackendCompatibility, QECBackend
from ai_qec.training import (
    ExecutionPlanner,
    ExecutionSpec,
    ResolvedExecutionPlan,
    Trainer,
    TrainingSpec,
)
from ai_qec.training.checkpoint import ModelCheckpoint, TrainingRecoveryCheckpoint

__all__ = [
    "AccuracyGate",
    "AccuracyGateSpec",
    "ArtifactKind",
    "ArtifactManifest",
    "ArtifactRef",
    "ArtifactRepository",
    "Attempt",
    "AttemptStatus",
    "BackendCompatibility",
    "DatasetArtifact",
    "DatasetIdentityProvider",
    "DatasetInstance",
    "DatasetKey",
    "DatasetRegistry",
    "DatasetResolution",
    "DatasetResolver",
    "DatasetRole",
    "DatasetSpec",
    "DatasetSplit",
    "DecodeRequest",
    "DecodeResult",
    "DecodeStatus",
    "Decoder",
    "DecoderEvaluation",
    "ExecutionSpec",
    "ExecutionPlanner",
    "Experiment",
    "ExperimentFactory",
    "ExperimentSpec",
    "GateDecision",
    "MetricEstimate",
    "ModelCheckpoint",
    "ModelSpec",
    "NoiseApproximation",
    "NoiseSpec",
    "NotebookExperiment",
    "NotebookPlatform",
    "NotebookRun",
    "QECBackend",
    "QECBatch",
    "QECDataGenerator",
    "QECSpec",
    "RandomStreamDescriptor",
    "RandomStreams",
    "RecoveryPlan",
    "RecoverySource",
    "ResolvedExecutionPlan",
    "ScientificAcceptanceResult",
    "ScientificEvaluationResult",
    "ScientificEvaluationSpec",
    "ScientificEvaluator",
    "StageRecord",
    "StageRecorder",
    "StageStatus",
    "Trainer",
    "TrainingRecoveryCheckpoint",
    "TrainingSpec",
]
