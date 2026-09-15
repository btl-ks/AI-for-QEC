"""Trainer interface."""

from __future__ import annotations


class Trainer:
    """Base interface for experiment trainers."""

    def fit(self) -> dict[str, float]:
        """Train a model and return validation metrics."""
        raise NotImplementedError
