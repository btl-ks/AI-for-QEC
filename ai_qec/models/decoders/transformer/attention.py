"""Attention block placeholder.

The initial project slice uses detector-summary features and a linear baseline.
This module exists so the public API matches the design document.
"""

from __future__ import annotations


class AttentionBlock:
    """Named placeholder for future attention block implementations."""

    def __init__(self, name: str = "placeholder") -> None:
        """Store a block name for debugging and registry compatibility."""
        self.name = name
