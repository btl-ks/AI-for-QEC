"""Opaque dataset identity contract.

The canonicalization and hashing algorithm is intentionally not implemented.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .spec import DatasetSpec


@dataclass(frozen=True, slots=True)
class DatasetKey:
    """Content identity derived from DatasetSpec generation semantics."""

    value: str
    algorithm: str
    schema_version: str = "dataset-key-v1"


@runtime_checkable
class DatasetIdentityProvider(Protocol):
    """Canonicalize DatasetSpec generation semantics into an opaque key."""

    def key_for(self, spec: DatasetSpec) -> DatasetKey: ...
