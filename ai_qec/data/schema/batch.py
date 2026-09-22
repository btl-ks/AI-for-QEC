"""Unified batch passed from QEC backends to trainers and decoders."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, Mapping, TypeVar


ArrayT = TypeVar("ArrayT")


class BatchRepresentation(StrEnum):
    PYTORCH_TENSOR = "pytorch-tensor"
    NUMPY_ARRAY = "numpy-array"
    BIT_PACKED_CPU_BUFFER = "bit-packed-cpu-buffer"
    PYTORCH_CUDA_TENSOR = "pytorch-cuda-tensor"
    DLPACK = "dlpack"


class MemoryResidency(StrEnum):
    HOST = "host"
    PINNED_HOST = "pinned-host"
    CUDA_DEVICE = "cuda-device"


@dataclass(frozen=True, slots=True)
class BatchLayout:
    """Runtime representation and residency evidence for one QECBatch."""

    representation: BatchRepresentation
    residency: MemoryResidency
    device: str
    dtype: str
    shape: tuple[int, ...]
    interchange: str | None = None
    zero_copy: bool = False


@dataclass(frozen=True, slots=True)
class QECBatch(Generic[ArrayT]):
    """Backend-neutral QEC samples.

    ``ArrayT`` can later be a NumPy, PyTorch, CuPy, or another array type.
    """

    detector_events: ArrayT
    observable_truth: ArrayT
    sample_ids: tuple[str, ...]
    dataset_artifact_id: str
    layout: BatchLayout
    schema_version: str = "qec-batch-v1"
    measurements: ArrayT | None = None
    physical_errors: ArrayT | None = None
    detector_coordinates: ArrayT | None = None
    round_mask: ArrayT | None = None
    context: Mapping[str, object] = field(default_factory=dict)
