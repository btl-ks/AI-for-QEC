"""Vendor-neutral validated noise models selected by ``noise.family``."""

from dataclasses import dataclass

from ai_qec.registry.catalog import NOISE
from ai_qec.registry.validation import ConfigurationError

from .noise import NoiseSpec


@dataclass(frozen=True, slots=True)
class IndependentPhaseFlipNoise:
    """Each data qubit independently suffers ``Z`` with ``physical_error_rate``."""

    physical_error_rate: float


@NOISE.register("independent-phase-flip")
def build_independent_phase_flip(*, spec: NoiseSpec) -> IndependentPhaseFlipNoise:
    if spec.family != "independent-phase-flip":
        raise ConfigurationError(
            f"[noise.family] expected 'independent-phase-flip', got {spec.family!r}"
        )
    if spec.time_dependent:
        raise ConfigurationError(
            "[noise.time_dependent] independent-phase-flip noise is time independent"
        )
    unknown = sorted(set(spec.parameters) - {"physical_error_rate"})
    if unknown:
        raise ConfigurationError(
            f"[noise.parameters] unknown keys for independent-phase-flip: {unknown}"
        )
    if "physical_error_rate" not in spec.parameters:
        raise ConfigurationError("[noise.parameters.physical_error_rate] is required")
    rate = spec.parameters["physical_error_rate"]
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not 0.0 <= float(rate) <= 0.5
    ):
        raise ConfigurationError(
            f"[noise.parameters.physical_error_rate] must be a number in [0, 0.5], got {rate!r}"
        )
    return IndependentPhaseFlipNoise(physical_error_rate=float(rate))
