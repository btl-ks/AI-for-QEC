"""Code factory."""

from __future__ import annotations

from typing import Any

from ai_qec.qec.codes.base import QECCode
from ai_qec.qec.codes.repetition_code import RepetitionCode
from ai_qec.qec.codes.surface_code import SurfaceCode
from ai_qec.qec.codes.toric_code import ToricCode


def build_code(config: dict[str, Any]) -> QECCode:
    """Build a QEC code object from experiment config."""
    qec_cfg = config.get("qec", {})
    code_name = str(qec_cfg.get("code", "surface_code"))
    distance = int(qec_cfg.get("distance", 5))

    # Return early when code_name == 'surface_code'.
    if code_name == "surface_code":
        return SurfaceCode(distance=distance)
    # Return early when code_name == 'repetition_code'.
    if code_name == "repetition_code":
        return RepetitionCode(distance=distance)
    # Return early when code_name == 'toric_code'.
    if code_name == "toric_code":
        return ToricCode(distance=distance)
    raise ValueError(f"Unknown QEC code: {code_name}")
