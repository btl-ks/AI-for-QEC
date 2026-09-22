"""Canonical configuration-field-to-registry declarations."""

from collections.abc import Mapping
from types import MappingProxyType

from ai_qec.registry import Registry


CODES = Registry[object]("qec.code_family")
CIRCUITS = Registry[object]("qec.circuit_adapter")
NOISE = Registry[object]("noise.family")
NOISE_ADAPTERS = Registry[object]("noise.adapter")
GENERATORS = Registry[object]("dataset.generator")
MODELS = Registry[object]("model.family")
OPTIMIZERS = Registry[object]("training.optimizer")
SCHEDULERS = Registry[object]("training.scheduler")
LOSSES = Registry[object]("training.loss")
TRAINERS = Registry[object]("execution.trainer_framework")
DECODERS = Registry[object]("scientific_evaluation.baseline_decoders")
CPU_TO_GPU_PIPELINES = Registry[object]("data_pipeline.cpu_to_gpu.technology")
GPU_TO_GPU_PIPELINES = Registry[object]("data_pipeline.gpu_to_gpu.technology")


REGISTRIES_BY_PATH: Mapping[str, Registry[object]] = MappingProxyType(
    {
        CODES.name: CODES,
        CIRCUITS.name: CIRCUITS,
        NOISE.name: NOISE,
        NOISE_ADAPTERS.name: NOISE_ADAPTERS,
        GENERATORS.name: GENERATORS,
        MODELS.name: MODELS,
        OPTIMIZERS.name: OPTIMIZERS,
        SCHEDULERS.name: SCHEDULERS,
        LOSSES.name: LOSSES,
        TRAINERS.name: TRAINERS,
        DECODERS.name: DECODERS,
        CPU_TO_GPU_PIPELINES.name: CPU_TO_GPU_PIPELINES,
        GPU_TO_GPU_PIPELINES.name: GPU_TO_GPU_PIPELINES,
    }
)
