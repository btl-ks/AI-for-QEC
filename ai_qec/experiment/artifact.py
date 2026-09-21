"""Generic immutable artifact references and manifests."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Protocol, runtime_checkable


class ArtifactKind(StrEnum):
    DATASET = "dataset"
    MODEL_CHECKPOINT = "model-checkpoint"
    TRAINING_RECOVERY_CHECKPOINT = "training-recovery-checkpoint"
    PREDICTIONS = "predictions"
    METRICS = "metrics"
    ACCURACY_GATE = "accuracy-gate"
    REPORT = "report"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Stable pointer to an immutable artifact."""

    artifact_id: str
    kind: ArtifactKind
    uri: str
    checksum: str
    media_type: str


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Auditable producer and metadata record for an ArtifactRef."""

    artifact: ArtifactRef
    producer_attempt_id: str
    producer_stage_id: str
    created_at: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "artifact-manifest-v1"


@runtime_checkable
class ArtifactRepository(Protocol):
    """Commit and verify immutable artifacts without fixing a storage backend."""

    def commit(self, manifest: ArtifactManifest) -> ArtifactRef: ...

    def verify(self, artifact: ArtifactRef) -> bool: ...

    def manifest_for(self, artifact_id: str) -> ArtifactManifest | None: ...
