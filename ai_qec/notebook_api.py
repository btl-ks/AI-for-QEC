"""Stable public facade for notebooks and external experiment drivers.

Importing this module loads only contracts and dependency-free helpers.
``LocalNotebookPlatform`` loads the executable implementations (Stim, PyTorch,
PyMatching) when it is constructed.
"""

from ai_qec.registry.validation import (
    ConfigurationError,
    MissingConfigurationError,
    UnresolvedConfigurationError,
    build_from_config,
    find_unresolved,
    get_config_path,
    validate_config,
    validate_no_unresolved,
    validate_registered_selections,
)
from ai_qec.data.datasets.local import DatasetIntegrityError
from ai_qec.data.datasets.artifact import DatasetArtifact, DatasetSplit
from ai_qec.data.datasets.identity import DatasetIdentityProvider, DatasetKey
from ai_qec.data.datasets.instance import DatasetInstance, DatasetRole
from ai_qec.data.datasets.registry import DatasetRegistry
from ai_qec.data.datasets.resolution import DatasetResolution, DatasetResolver
from ai_qec.data.datasets.spec import DatasetSpec
from ai_qec.data.generators.protocol import (
    GeneratorDevice,
    QECDataGenerator,
    SyndromeGeneratorDescriptor,
)
from ai_qec.data.pipeline import (
    CPUToGPUDataPipeline,
    CPUToGPUPipelineSpec,
    GPUToGPUDataPipeline,
    GPUToGPUPipelineSpec,
    TransferEvidence,
)
from ai_qec.data.schema.batch import BatchLayout, BatchRepresentation, MemoryResidency, QECBatch
from ai_qec.evaluation.scientific.accuracy_gate import AccuracyGate
from ai_qec.evaluation.scientific.evaluator import ScientificEvaluator
from ai_qec.evaluation.scientific.local import AccuracyGateError, EvaluationError
from ai_qec.evaluation.scientific.result import (
    DecoderEvaluation,
    GateDecision,
    MetricEstimate,
    ScientificAcceptanceResult,
    ScientificEvaluationResult,
)
from ai_qec.evaluation.scientific.spec import AccuracyGateSpec, ScientificEvaluationSpec
from ai_qec.evaluation.scientific.statistics import paired_difference_interval, wilson_interval
from ai_qec.experiment.artifact import (
    ArtifactKind,
    ArtifactManifest,
    ArtifactRef,
    ArtifactRepository,
)
from ai_qec.experiment.grid import GridPoint, config_grid, with_overrides
from ai_qec.experiment.local import AttemptInProgressError, AttemptStateError
from ai_qec.experiment.random_streams import RandomStreamDescriptor, RandomStreams
from ai_qec.experiment.recovery import RecoveryPlan, RecoverySource
from ai_qec.experiment.run import Attempt, AttemptStatus, Experiment, ExperimentFactory
from ai_qec.experiment.spec import ExperimentSpec
from ai_qec.experiment.stage import StageRecord, StageRecorder, StageStatus
from ai_qec.models.decoders.protocol import (
    DecodeRequest,
    DecodeResult,
    DecodeStatus,
    Decoder,
    DecoderRuntimeDescriptor,
)
from ai_qec.models.spec import ModelSpec
from ai_qec.paper.local_runtime import LocalNotebookPlatform
from ai_qec.paper.protocol import NotebookExperiment, NotebookPlatform, NotebookRun
from ai_qec.qec.backends.protocol import BackendCompatibility, QECBackend
from ai_qec.qec.circuits.protocol import CircuitBuildResult, QECCircuitAdapter
from ai_qec.qec.noise import NoiseApproximation, NoiseCompilation, NoiseCompiler, NoiseSpec
from ai_qec.qec.spec import QECSpec
from ai_qec.registry.catalog import (
    CIRCUITS,
    CODES,
    CPU_TO_GPU_PIPELINES,
    DECODERS,
    GENERATORS,
    GPU_TO_GPU_PIPELINES,
    LOSSES,
    MODELS,
    NOISE,
    NOISE_ADAPTERS,
    OPTIMIZERS,
    REGISTRIES_BY_PATH,
    SCHEDULERS,
    TRAINERS,
    TRAINING_STEP_EXECUTORS,
)
from ai_qec.registry.core import (
    DuplicateRegistrationError,
    Registry,
    RegistryError,
    UnknownRegistrationError,
)
from ai_qec.reporting.figures import (
    SweepPoint,
    figure_png,
    plot_decoder_comparison,
    plot_failure_rate_sweep,
    plot_logical_class_histograms,
    plot_logical_classes,
    plot_training_history,
)
from ai_qec.technology import TechnologyId, TechnologyImplementation
from ai_qec.training.checkpoint.model import ModelCheckpoint
from ai_qec.training.checkpoint.recovery import TrainingRecoveryCheckpoint
from ai_qec.training.execution import ExecutionPlanner, ExecutionSpec, ResolvedExecutionPlan
from ai_qec.training.execution_planner import ExecutionConfigurationError
from ai_qec.training.executors.protocol import (
    BatchSignature,
    BatchSignatureEvidence,
    TensorSignature,
    TrainingStepContext,
    TrainingStepEvidence,
    TrainingStepExecutor,
    TrainingStepPlan,
    TrainingStepResult,
)
from ai_qec.training.spec import (
    Trainer,
    TrainingSpec,
)
from ai_qec.utils.paths import find_project_root

__all__ = [
    "AccuracyGate",
    "AccuracyGateError",
    "AttemptInProgressError",
    "AttemptStateError",
    "DatasetIntegrityError",
    "EvaluationError",
    "ExecutionConfigurationError",
    "LocalNotebookPlatform",
    "SweepPoint",
    "config_grid",
    "figure_png",
    "find_project_root",
    "GridPoint",
    "with_overrides",
    "paired_difference_interval",
    "plot_decoder_comparison",
    "plot_failure_rate_sweep",
    "plot_logical_class_histograms",
    "plot_logical_classes",
    "plot_training_history",
    "wilson_interval",
    "AccuracyGateSpec",
    "ArtifactKind",
    "ArtifactManifest",
    "ArtifactRef",
    "ArtifactRepository",
    "Attempt",
    "AttemptStatus",
    "BackendCompatibility",
    "BatchLayout",
    "BatchRepresentation",
    "BatchSignature",
    "BatchSignatureEvidence",
    "CIRCUITS",
    "CODES",
    "CPUToGPUDataPipeline",
    "CPUToGPUPipelineSpec",
    "CPU_TO_GPU_PIPELINES",
    "CircuitBuildResult",
    "ConfigurationError",
    "DECODERS",
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
    "DecoderRuntimeDescriptor",
    "DuplicateRegistrationError",
    "ExecutionSpec",
    "ExecutionPlanner",
    "Experiment",
    "ExperimentFactory",
    "ExperimentSpec",
    "GateDecision",
    "GENERATORS",
    "GPUToGPUDataPipeline",
    "GPUToGPUPipelineSpec",
    "GPU_TO_GPU_PIPELINES",
    "GeneratorDevice",
    "LOSSES",
    "MODELS",
    "MetricEstimate",
    "MissingConfigurationError",
    "ModelCheckpoint",
    "ModelSpec",
    "NoiseApproximation",
    "NoiseCompilation",
    "NoiseCompiler",
    "NoiseSpec",
    "NOISE",
    "NOISE_ADAPTERS",
    "NotebookExperiment",
    "NotebookPlatform",
    "NotebookRun",
    "OPTIMIZERS",
    "QECBackend",
    "QECBatch",
    "QECDataGenerator",
    "QECCircuitAdapter",
    "QECSpec",
    "RandomStreamDescriptor",
    "RandomStreams",
    "REGISTRIES_BY_PATH",
    "Registry",
    "RegistryError",
    "RecoveryPlan",
    "RecoverySource",
    "ResolvedExecutionPlan",
    "ScientificAcceptanceResult",
    "ScientificEvaluationResult",
    "ScientificEvaluationSpec",
    "ScientificEvaluator",
    "SCHEDULERS",
    "StageRecord",
    "StageRecorder",
    "StageStatus",
    "SyndromeGeneratorDescriptor",
    "TechnologyId",
    "TechnologyImplementation",
    "TensorSignature",
    "TRAINERS",
    "TRAINING_STEP_EXECUTORS",
    "Trainer",
    "TransferEvidence",
    "TrainingRecoveryCheckpoint",
    "TrainingSpec",
    "TrainingStepContext",
    "TrainingStepEvidence",
    "TrainingStepExecutor",
    "TrainingStepPlan",
    "TrainingStepResult",
    "UnknownRegistrationError",
    "UnresolvedConfigurationError",
    "build_from_config",
    "find_unresolved",
    "get_config_path",
    "validate_config",
    "validate_no_unresolved",
    "validate_registered_selections",
    "MemoryResidency",
]
