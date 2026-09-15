"""Canonical QEC sample schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QECSample:
    """One canonical sample joining detector data, labels, and contexts."""
    syndrome: np.ndarray | None
    detector_events: np.ndarray | None
    logical_label: int
    physical_error: np.ndarray | None = None
    code_context: dict[str, Any] = field(default_factory=dict)
    noise_context: dict[str, Any] = field(default_factory=dict)
    circuit_context: dict[str, Any] = field(default_factory=dict)
    device_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
