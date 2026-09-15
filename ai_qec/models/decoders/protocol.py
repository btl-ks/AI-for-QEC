"""Project-local decoder contract for code-capacity recovery chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class DecodeRequest:
    syndrome: np.ndarray
    p_error: float | None = None


@dataclass(frozen=True)
class DecodeResult:
    recovery: np.ndarray | None
    success: bool
    steps: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def timed_out(self) -> bool:
        return not self.success


class Decoder(Protocol):
    def decode(self, request: DecodeRequest, *, rng: np.random.Generator | None = None) -> DecodeResult:
        """Return a recovery chain or an explicit failure with diagnostics."""
        ...
