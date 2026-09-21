"""Named deterministic random stream contracts."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RandomStreamDescriptor:
    """Stable identity for one independent random stream."""

    name: str
    master_seed: int
    derived_seed: int
    algorithm: str


@runtime_checkable
class RandomStreams(Protocol):
    """Derive and restore independent streams without exposing RNG libraries."""

    def describe(self, name: str) -> RandomStreamDescriptor: ...

    def snapshot(self, name: str) -> bytes: ...

    def restore(self, name: str, state: bytes) -> None: ...
