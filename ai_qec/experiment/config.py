"""Strict translation of an experiment configuration mapping into contract specs."""

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import math

from ai_qec.config_validation import ConfigurationError, MissingConfigurationError
from ai_qec.data.datasets.spec import DatasetSpec
from ai_qec.data.pipeline import CPUToGPUPipelineSpec
from ai_qec.evaluation.scientific.spec import AccuracyGateSpec, ScientificEvaluationSpec
from ai_qec.models.spec import ModelSpec
from ai_qec.qec.noise import NoiseSpec
from ai_qec.qec.spec import QECSpec
from ai_qec.training.execution import ExecutionSpec
from ai_qec.training.spec import TrainingSpec
from ai_qec.utils.hashing import sha256_json, to_jsonable

from .spec import ExperimentSpec

SCHEMA_VERSION = "0.1"
TECHNOLOGY_PROFILE = "v0-1"
ALLOWED_KEYS: Mapping[str, frozenset[str]] = {
    "experiment": frozenset({"name", "master_seed"}),
    "qec": frozenset(
        {"code_family", "distance", "rounds", "logical_basis", "circuit_family", "circuit_adapter"}
    ),
    "noise": frozenset({"family", "adapter", "parameters", "time_dependent"}),
    "dataset": frozenset(
        {
            "generator",
            "generator_version",
            "backend_semantics",
            "train_samples",
            "validation_samples",
            "test_samples",
            "seed",
            "split_policy",
            "schema_version",
            "preprocessing",
        }
    ),
    "model": frozenset({"family", "architecture_version", "parameters", "decoding"}),
    "training": frozenset(
        {
            "optimizer",
            "optimizer_parameters",
            "learning_rate",
            "epochs",
            "batch_size",
            "scheduler",
            "loss",
            "loss_parameters",
            "checkpoint_policy",
        }
    ),
    "execution": frozenset(
        {
            "trainer_framework",
            "device",
            "cpu_count",
            "gpu_count",
            "distributed",
            "num_workers",
            "mixed_precision",
            "compile_model",
        }
    ),
    "data_pipeline": frozenset({"cpu_to_gpu"}),
    "scientific_evaluation": frozenset(
        {
            "baseline_decoders",
            "primary_metric",
            "confidence_level",
            "stopping_rule",
            "invalid_sample_policy",
        }
    ),
    "accuracy_gate": frozenset(
        {
            "gate_id",
            "baseline_decoder",
            "primary_metric",
            "comparison_rule",
            "tolerance",
            "confidence_level",
            "stage",
        }
    ),
    "performance": frozenset({"shots", "repetitions", "warmup"}),
}
TOP_LEVEL = frozenset({"schema_version", "technology_profile", *ALLOWED_KEYS})
REQUIRED_SELECTIONS = (
    "qec.code_family",
    "noise.family",
    "noise.adapter",
    "dataset.generator",
    "model.family",
    "training.optimizer",
    "training.scheduler",
    "training.loss",
    "execution.trainer_framework",
    "scientific_evaluation.baseline_decoders",
    "data_pipeline.cpu_to_gpu.technology",
)
PIPELINE_KEYS = frozenset(
    {"technology", "pin_memory", "non_blocking", "prefetch_factor", "persistent_workers"}
)


@dataclass(frozen=True, slots=True)
class ExperimentConfiguration:
    """Validated specs plus the runtime settings that have no contract type yet."""

    spec: ExperimentSpec
    decoding: Mapping[str, object]
    pipeline: CPUToGPUPipelineSpec
    performance: Mapping[str, int]
    config: Mapping[str, object]
    digest: str


def _section(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    if name not in config:
        raise MissingConfigurationError((name,))
    value = config[name]
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"[{name}] must be a mapping")
    unknown = sorted(set(value) - ALLOWED_KEYS[name])
    if unknown:
        raise ConfigurationError(f"[{name}] unknown keys: {unknown}")
    return value


def _get(section: Mapping[str, object], path: str, key: str, kind, *, default=None, required=True):
    if key not in section:
        if required:
            raise MissingConfigurationError((f"{path}.{key}",))
        return default
    value = section[key]
    if kind is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind is float:
        ok = (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
        value = float(value) if ok else value
    elif kind is bool:
        ok = isinstance(value, bool)
    elif kind is str:
        ok = isinstance(value, str) and bool(value)
    elif kind is Mapping:
        ok = isinstance(value, Mapping)
    else:
        ok = isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    if not ok:
        raise ConfigurationError(f"[{path}.{key}] has invalid value {value!r}")
    return value


def _positive(value: int, path: str) -> int:
    if value < 1:
        raise ConfigurationError(f"[{path}] must be >= 1, got {value}")
    return value


def _choice(value: str, path: str, allowed: Sequence[str]) -> str:
    if value not in allowed:
        raise ConfigurationError(f"[{path}] must be one of {list(allowed)}, got {value!r}")
    return value


def parse_experiment_config(config: Mapping[str, object]) -> ExperimentConfiguration:
    """Validate keys, types and supported values; build all contract specs."""

    unknown = sorted(set(config) - TOP_LEVEL)
    if unknown:
        raise ConfigurationError(f"unknown top-level configuration keys: {unknown}")
    _choice(_get(config, "", "schema_version", str).strip(), "schema_version", (SCHEMA_VERSION,))
    _choice(
        _get(config, "", "technology_profile", str), "technology_profile", (TECHNOLOGY_PROFILE,)
    )

    experiment = _section(config, "experiment")
    qec = _section(config, "qec")
    noise = _section(config, "noise")
    dataset = _section(config, "dataset")
    model = _section(config, "model")
    training = _section(config, "training")
    execution = _section(config, "execution")
    pipeline = _section(config, "data_pipeline")
    evaluation = _section(config, "scientific_evaluation")
    gate = _section(config, "accuracy_gate")
    performance = _section(config, "performance") if "performance" in config else {}

    qec_spec = QECSpec(
        code_family=_get(qec, "qec", "code_family", str),
        distance=_positive(_get(qec, "qec", "distance", int), "qec.distance"),
        rounds=_positive(_get(qec, "qec", "rounds", int), "qec.rounds"),
        logical_basis=_get(qec, "qec", "logical_basis", str),
        circuit_family=_get(qec, "qec", "circuit_family", str),
    )
    noise_spec = NoiseSpec(
        family=_get(noise, "noise", "family", str),
        parameters=dict(_get(noise, "noise", "parameters", Mapping)),
        time_dependent=_get(noise, "noise", "time_dependent", bool, default=False, required=False),
    )
    dataset_spec = DatasetSpec(
        qec=qec_spec,
        noise=noise_spec,
        generator=_get(dataset, "dataset", "generator", str),
        generator_version=_get(dataset, "dataset", "generator_version", str),
        backend_semantics=_get(dataset, "dataset", "backend_semantics", str),
        train_samples=_positive(
            _get(dataset, "dataset", "train_samples", int), "dataset.train_samples"
        ),
        validation_samples=_positive(
            _get(dataset, "dataset", "validation_samples", int), "dataset.validation_samples"
        ),
        test_samples=_positive(
            _get(dataset, "dataset", "test_samples", int), "dataset.test_samples"
        ),
        seed=_get(dataset, "dataset", "seed", int),
        split_policy=_get(dataset, "dataset", "split_policy", str),
        schema_version=_get(dataset, "dataset", "schema_version", str),
        preprocessing=tuple(
            _get(dataset, "dataset", "preprocessing", list, default=(), required=False)
        ),
    )
    model_spec = ModelSpec(
        family=_get(model, "model", "family", str),
        architecture_version=_get(model, "model", "architecture_version", str),
        parameters=dict(_get(model, "model", "parameters", Mapping)),
    )
    training_spec = TrainingSpec(
        optimizer=_get(training, "training", "optimizer", str),
        learning_rate=_get(training, "training", "learning_rate", float),
        epochs=_positive(_get(training, "training", "epochs", int), "training.epochs"),
        batch_size=_positive(_get(training, "training", "batch_size", int), "training.batch_size"),
        scheduler=_get(training, "training", "scheduler", str),
        loss=_get(training, "training", "loss", str),
        checkpoint_policy=_choice(
            _get(
                training,
                "training",
                "checkpoint_policy",
                str,
                default="epoch-boundary",
                required=False,
            ),
            "training.checkpoint_policy",
            ("epoch-boundary",),
        ),
        optimizer_parameters=dict(
            _get(training, "training", "optimizer_parameters", Mapping, default={}, required=False)
        ),
        loss_parameters=dict(
            _get(training, "training", "loss_parameters", Mapping, default={}, required=False)
        ),
    )
    execution_spec = ExecutionSpec(
        device=_get(execution, "execution", "device", str),
        cpu_count=_get(execution, "execution", "cpu_count", int),
        gpu_count=_get(execution, "execution", "gpu_count", int),
        distributed=_get(execution, "execution", "distributed", bool),
        num_workers=_get(execution, "execution", "num_workers", int),
        mixed_precision=_get(execution, "execution", "mixed_precision", bool),
        compile_model=_get(execution, "execution", "compile_model", bool),
        trainer_framework=_get(execution, "execution", "trainer_framework", str),
    )
    evaluation_spec = ScientificEvaluationSpec(
        baseline_decoders=tuple(
            _get(evaluation, "scientific_evaluation", "baseline_decoders", list)
        ),
        primary_metric=_choice(
            _get(evaluation, "scientific_evaluation", "primary_metric", str),
            "scientific_evaluation.primary_metric",
            ("logical_error_rate",),
        ),
        confidence_level=_get(evaluation, "scientific_evaluation", "confidence_level", float),
        stopping_rule=_choice(
            _get(evaluation, "scientific_evaluation", "stopping_rule", str),
            "scientific_evaluation.stopping_rule",
            ("fixed-shots",),
        ),
        invalid_sample_policy=_choice(
            _get(evaluation, "scientific_evaluation", "invalid_sample_policy", str),
            "scientific_evaluation.invalid_sample_policy",
            ("count-as-failure", "fail"),
        ),
        noise_points=(float(noise_spec.parameters.get("physical_error_rate", math.nan)),),
        code_distances=(qec_spec.distance,),
    )
    if not evaluation_spec.baseline_decoders:
        raise ConfigurationError(
            "[scientific_evaluation.baseline_decoders] must name at least one baseline"
        )
    gate_spec = AccuracyGateSpec(
        gate_id=_get(gate, "accuracy_gate", "gate_id", str),
        baseline_decoder=_get(gate, "accuracy_gate", "baseline_decoder", str),
        primary_metric=_choice(
            _get(gate, "accuracy_gate", "primary_metric", str),
            "accuracy_gate.primary_metric",
            ("logical_error_rate",),
        ),
        comparison_rule=_choice(
            _get(gate, "accuracy_gate", "comparison_rule", str),
            "accuracy_gate.comparison_rule",
            ("paired-non-inferiority", "paired-relative-non-inferiority"),
        ),
        tolerance=_get(gate, "accuracy_gate", "tolerance", float),
        confidence_level=_get(gate, "accuracy_gate", "confidence_level", float),
        stage=_choice(
            _get(gate, "accuracy_gate", "stage", str, default="gate-a", required=False),
            "accuracy_gate.stage",
            ("gate-a",),
        ),
    )
    for path, level in (
        ("scientific_evaluation.confidence_level", evaluation_spec.confidence_level),
        ("accuracy_gate.confidence_level", gate_spec.confidence_level),
    ):
        if not 0.0 < level < 1.0:
            raise ConfigurationError(f"[{path}] must be in (0, 1), got {level}")
    if gate_spec.comparison_rule == "paired-relative-non-inferiority" and gate_spec.tolerance < 0:
        raise ConfigurationError(
            f"[accuracy_gate.tolerance] is a relative tolerance for {gate_spec.comparison_rule} "
            f"and must be >= 0, got {gate_spec.tolerance}"
        )
    if gate_spec.baseline_decoder not in evaluation_spec.baseline_decoders:
        raise ConfigurationError(
            f"[accuracy_gate.baseline_decoder] {gate_spec.baseline_decoder!r} is not one of the evaluated baselines"
        )

    cpu_to_gpu = _get(pipeline, "data_pipeline", "cpu_to_gpu", Mapping)
    unknown = sorted(set(cpu_to_gpu) - PIPELINE_KEYS)
    if unknown:
        raise ConfigurationError(f"[data_pipeline.cpu_to_gpu] unknown keys: {unknown}")
    pipeline_spec = CPUToGPUPipelineSpec(
        technology_id=_get(cpu_to_gpu, "data_pipeline.cpu_to_gpu", "technology", str),
        pin_memory=_get(cpu_to_gpu, "data_pipeline.cpu_to_gpu", "pin_memory", bool),
        non_blocking=_get(cpu_to_gpu, "data_pipeline.cpu_to_gpu", "non_blocking", bool),
        prefetch_factor=_get(
            cpu_to_gpu,
            "data_pipeline.cpu_to_gpu",
            "prefetch_factor",
            int,
            default=2,
            required=False,
        ),
        num_workers=execution_spec.num_workers,
        persistent_workers=_get(
            cpu_to_gpu,
            "data_pipeline.cpu_to_gpu",
            "persistent_workers",
            bool,
            default=False,
            required=False,
        ),
    )
    performance_settings = {
        "shots": _positive(
            _get(performance, "performance", "shots", int, default=1000, required=False),
            "performance.shots",
        ),
        "repetitions": _positive(
            _get(performance, "performance", "repetitions", int, default=3, required=False),
            "performance.repetitions",
        ),
        "warmup": _get(performance, "performance", "warmup", int, default=1, required=False),
    }

    spec = ExperimentSpec(
        experiment_name=_get(experiment, "experiment", "name", str),
        dataset=dataset_spec,
        model=model_spec,
        training=training_spec,
        execution=execution_spec,
        scientific_evaluation=evaluation_spec,
        accuracy_gate=gate_spec,
        master_seed=_get(experiment, "experiment", "master_seed", int),
    )
    snapshot = copy.deepcopy(to_jsonable(config))
    return ExperimentConfiguration(
        spec=spec,
        decoding=dict(_get(model, "model", "decoding", Mapping)),
        pipeline=pipeline_spec,
        performance=performance_settings,
        config=snapshot,
        digest=sha256_json(snapshot),
    )
