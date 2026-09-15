"""Dataset manifest schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    """Serializable provenance record for one generated dataset."""
    dataset_id: str
    generator: str
    sample_counts: dict[str, int]
    feature_names: list[str]
    config_hash: str
    generation_hash: str
    schema_version: int = 1
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    effective_generation: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass manifest into JSON-ready dictionaries."""
        return asdict(self)
