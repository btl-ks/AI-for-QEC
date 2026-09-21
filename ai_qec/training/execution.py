"""Execution resource specification separated from learning semantics."""

from dataclasses import dataclass
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
    schema_version: str = "execution-spec-v1"


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


@runtime_checkable
class ExecutionPlanner(Protocol):
    """Validate requested resources without creating runtime side effects."""

    def resolve(self, spec: ExecutionSpec) -> ResolvedExecutionPlan: ...
