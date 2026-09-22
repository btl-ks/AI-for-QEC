"""Backend-neutral decoder request and result protocol."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, Mapping, Protocol, TypeVar, runtime_checkable

from ai_qec.data.schema.batch import QECBatch


ArrayT = TypeVar("ArrayT")


class DecodeStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed-out"
    NOT_CONVERGED = "not-converged"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class DecoderRuntimeDescriptor:
    """Actual decoder technology used for a DecodeResult."""

    technology_id: str
    technology_version: str
    device: str


@dataclass(frozen=True, slots=True)
class DecodeRequest(Generic[ArrayT]):
    """A decoder invocation over an explicitly identified QEC batch."""

    request_id: str
    decoder_id: str
    batch: QECBatch[ArrayT]
    context: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecodeResult(Generic[ArrayT]):
    """Predictions and explicit non-success state from a decoder."""

    request_id: str
    decoder_id: str
    status: DecodeStatus
    predictions: ArrayT | None
    runtime: DecoderRuntimeDescriptor
    failed_sample_ids: tuple[str, ...] = ()
    error: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class Decoder(Protocol[ArrayT]):
    """Common behavior implemented by AI, classical, and runtime decoders."""

    @property
    def decoder_id(self) -> str: ...

    def decode(self, request: DecodeRequest[ArrayT]) -> DecodeResult[ArrayT]: ...
