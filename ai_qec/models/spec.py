"""Declarative model specification."""

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Scientific model identity without framework-specific model objects."""

    family: str
    architecture_version: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "model-spec-v1"
