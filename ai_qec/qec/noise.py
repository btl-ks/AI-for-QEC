"""Vendor-neutral noise specification contracts."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class NoiseApproximation(StrEnum):
    """Whether a backend can preserve requested noise semantics."""

    EXACT = "exact"
    EXPLICIT_APPROXIMATION = "explicit-approximation"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    """Declarative physical noise model independent of a simulator."""

    family: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    time_dependent: bool = False
    schema_version: str = "noise-spec-v1"
