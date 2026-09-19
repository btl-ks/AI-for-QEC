"""Noise factory."""

from __future__ import annotations

from typing import Any

from ai_qec.qec.noise.base import NoiseModel
from ai_qec.qec.noise.crosstalk import WeakCrosstalkNoise
from ai_qec.qec.noise.phase_flip import IndependentPhaseFlipNoise


def build_noise_model(config: dict[str, Any]) -> NoiseModel:
    """Build a target-noise sampler from experiment config."""
    # Return early when str(config.get('noise', {}).get('model', '')) == 'phase_flip'.
    if str(config.get("noise", {}).get("model", "")) == "phase_flip":
        return IndependentPhaseFlipNoise(float(config["noise"]["p_error"]))
    target_name = str(config.get("noise", {}).get("target", {}).get("name", "crosstalk"))
    # Return early when target_name == 'crosstalk'.
    if target_name == "crosstalk":
        return WeakCrosstalkNoise(config)
    raise ValueError(f"Unknown target noise model: {target_name}")
