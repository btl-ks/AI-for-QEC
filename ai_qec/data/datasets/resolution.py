"""Dataset resolution interface."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .artifact import DatasetArtifact
from .instance import DatasetInstance
from .spec import DatasetSpec


@dataclass(frozen=True, slots=True)
class DatasetResolution:
    """Result of a verified cache hit or committed cache miss."""

    artifact: DatasetArtifact
    instance: DatasetInstance
    cache_hit: bool


@runtime_checkable
class DatasetResolver(Protocol):
    """Resolve DatasetSpec without exposing storage or generator details."""

    def resolve(self, spec: DatasetSpec, attempt_id: str) -> DatasetResolution: ...
