"""Context metadata schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QECContext:
    """Container for code, noise, circuit, and device metadata."""
    code: dict[str, Any] = field(default_factory=dict)
    noise: dict[str, Any] = field(default_factory=dict)
    circuit: dict[str, Any] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
