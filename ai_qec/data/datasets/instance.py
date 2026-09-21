"""Run-local dataset binding contract."""

from dataclasses import dataclass
from enum import StrEnum


class DatasetRole(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    SCIENTIFIC_EVALUATION = "scientific-evaluation"


@dataclass(frozen=True, slots=True)
class DatasetInstance:
    """How one Attempt uses an immutable DatasetArtifact."""

    instance_id: str
    attempt_id: str
    dataset_artifact_id: str
    roles: tuple[DatasetRole, ...]
    resolved_from_cache: bool
