"""Execution resource specification separated from learning semantics."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Where and how training runs, without changing dataset identity."""

    device: str
    cpu_count: int
    gpu_count: int
    distributed: bool
    num_workers: int
    mixed_precision: bool
    compile_model: bool
    trainer_framework: str = "pytorch"
    step_executor: str = "pytorch-eager"
    step_executor_options: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "execution-spec-v2"


@dataclass(frozen=True, slots=True)
class ResolvedExecutionPlan:
    """Validated resource plan persisted before an Attempt begins."""

    device: str
    world_size: int
    cpu_count: int
    gpu_count: int
    num_workers: int
    distributed_backend: str | None
    mixed_precision_mode: str | None
    compile_model: bool
    environment_digest: str
    trainer_framework: str = "pytorch"
    step_executor: str = "pytorch-eager"
    step_executor_version: str = "1"
    step_executor_options: Mapping[str, object] = field(default_factory=dict)
    step_executor_options_digest: str = ""


@runtime_checkable
class ExecutionPlanner(Protocol):
    """Validate requested resources without creating runtime side effects."""

    def resolve(self, spec: ExecutionSpec) -> ResolvedExecutionPlan: ...
