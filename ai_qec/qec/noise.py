"""Vendor-neutral noise specification contracts."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, Mapping, Protocol, TypeVar, runtime_checkable


NoiseModelT = TypeVar("NoiseModelT")


class NoiseApproximation(StrEnum):
    """Whether a backend can preserve requested noise semantics."""

    EXACT = "exact"
    EXPLICIT_APPROXIMATION = "explicit-approximation"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    """Declarative physical noise model independent of a simulator."""

    family: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    time_dependent: bool = False
    schema_version: str = "noise-spec-v1"


@dataclass(frozen=True, slots=True)
class NoiseCompilation(Generic[NoiseModelT]):
    """Explicit translation of NoiseSpec into a selected backend model."""

    model: NoiseModelT | None
    technology_id: str
    technology_version: str
    source_spec_digest: str
    support: NoiseApproximation
    approximation_id: str | None = None
    reason: str | None = None


@runtime_checkable
class NoiseCompiler(Protocol[NoiseModelT]):
    """Compile NoiseSpec without silently changing its semantics."""

    @property
    def technology_id(self) -> str: ...

    def compile(self, spec: NoiseSpec) -> NoiseCompilation[NoiseModelT]: ...
