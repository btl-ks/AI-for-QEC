"""Adapter interface."""

from __future__ import annotations


class Adapter:
    """Base interface for model/domain adapters."""

    def adapt(self) -> None:
        """Apply an adaptation step in concrete adapter implementations."""
        raise NotImplementedError
