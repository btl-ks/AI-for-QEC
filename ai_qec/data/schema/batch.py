"""Unified batch passed from QEC backends to trainers and decoders."""

from dataclasses import dataclass, field
from typing import Generic, Mapping, TypeVar


ArrayT = TypeVar("ArrayT")


@dataclass(frozen=True, slots=True)
class QECBatch(Generic[ArrayT]):
    """Backend-neutral QEC samples.

    ``ArrayT`` can later be a NumPy, PyTorch, CuPy, or another array type.
    """

    detector_events: ArrayT
    observable_truth: ArrayT
    sample_ids: tuple[str, ...]
    dataset_artifact_id: str
    schema_version: str = "qec-batch-v1"
    measurements: ArrayT | None = None
    physical_errors: ArrayT | None = None
    detector_coordinates: ArrayT | None = None
    round_mask: ArrayT | None = None
    context: Mapping[str, object] = field(default_factory=dict)
