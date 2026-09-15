"""Reproducible random number helpers."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and NumPy RNGs, then return a fresh NumPy generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
