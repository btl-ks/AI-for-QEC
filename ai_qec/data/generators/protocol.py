"""QEC batch generator protocol."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable

from ai_qec.data.datasets.spec import DatasetSpec
from ai_qec.data.schema.batch import BatchRepresentation, MemoryResidency, QECBatch
from ai_qec.qec.backends.protocol import BackendCompatibility


ArrayT = TypeVar("ArrayT")


class GeneratorDevice(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class SyndromeGeneratorDescriptor:
    """Selected generator technology and expected output placement."""

    technology_id: str
    technology_version: str
    device: GeneratorDevice
    representation: BatchRepresentation
    residency: MemoryResidency


@runtime_checkable
class QECDataGenerator(Protocol[ArrayT]):
    """Generate backend-neutral batches for an accepted DatasetSpec."""

    @property
    def generator_id(self) -> str: ...

    @property
    def descriptor(self) -> SyndromeGeneratorDescriptor: ...

    def compatibility(self, spec: DatasetSpec) -> BackendCompatibility: ...

    def generate(self, spec: DatasetSpec) -> Iterable[QECBatch[ArrayT]]: ...
