"""Simulator interface."""

from __future__ import annotations

from typing import Any

import numpy as np


class QECBackend:
    """Base interface for QEC simulation or hardware sampling backends."""
    name = "base"

    def sample_batch(self, n_samples: int, rng: np.random.Generator) -> dict[str, Any]:
        """Generate one batch of features, labels, and sampled parameters."""
        raise NotImplementedError
