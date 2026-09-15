"""Paired counterfactual generator hook."""

from __future__ import annotations

from typing import Any


def generate_paired_dataset(config: dict[str, Any], output_dir: str) -> dict[str, Any]:
    """Fail explicitly: paired counterfactual data is not implemented."""
    raise NotImplementedError("Paired counterfactual generation is not implemented")
