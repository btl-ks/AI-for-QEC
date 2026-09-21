"""Immutable dataset artifact descriptors."""

from dataclasses import dataclass

from .identity import DatasetKey


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Manifest entry for one immutable dataset split."""

    name: str
    sample_count: int
    shard_uris: tuple[str, ...]
    checksum: str


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """A verified, immutable, reusable physical dataset."""

    artifact_id: str
    key: DatasetKey
    manifest_uri: str
    checksum: str
    splits: tuple[DatasetSplit, ...]
    generator_provenance_uri: str
    schema_version: str = "dataset-artifact-v1"
