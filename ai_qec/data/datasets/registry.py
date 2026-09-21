"""Registry interface; persistence is supplied by future implementations."""

from typing import Protocol, runtime_checkable

from .artifact import DatasetArtifact
from .identity import DatasetKey


@runtime_checkable
class DatasetRegistry(Protocol):
    """Lookup and atomically register immutable dataset artifacts."""

    def find(self, key: DatasetKey) -> DatasetArtifact | None: ...

    def register(self, artifact: DatasetArtifact) -> None: ...

    def verify(self, artifact: DatasetArtifact) -> bool: ...
