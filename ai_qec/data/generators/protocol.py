"""QEC batch generator protocol."""

from collections.abc import Iterable
from typing import Protocol, TypeVar, runtime_checkable

from ai_qec.data.datasets.spec import DatasetSpec
from ai_qec.data.schema.batch import QECBatch
from ai_qec.qec.backends.protocol import BackendCompatibility


ArrayT_co = TypeVar("ArrayT_co", covariant=True)


@runtime_checkable
class QECDataGenerator(Protocol[ArrayT_co]):
    """Generate backend-neutral batches for an accepted DatasetSpec."""

    @property
    def generator_id(self) -> str: ...

    def compatibility(self, spec: DatasetSpec) -> BackendCompatibility: ...

    def generate(self, spec: DatasetSpec) -> Iterable[QECBatch[ArrayT_co]]: ...
