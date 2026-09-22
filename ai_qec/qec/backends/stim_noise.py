"""Stim noise adapter: compiles NoiseSpec exactly or rejects it."""

from dataclasses import dataclass

from ai_qec.registries import NOISE_ADAPTERS
from ai_qec.technology import TechnologyId
from ai_qec.utils.hashing import sha256_json

from ..noise import NoiseApproximation, NoiseCompilation, NoiseSpec
from ..noise_models import build_independent_phase_flip


@dataclass(frozen=True, slots=True)
class StimPauliChannel:
    """One single-qubit Stim noise instruction applied to all data qubits."""

    instruction: str
    probability: float

    def append_to(self, circuit, targets) -> None:
        circuit.append(self.instruction, list(targets), self.probability)


class StimNoiseCompiler:
    technology_id = TechnologyId.STIM_NOISE.value

    def compile(self, spec: NoiseSpec) -> NoiseCompilation[StimPauliChannel]:
        import stim

        digest = sha256_json(spec)
        if spec.family != "independent-phase-flip" or spec.time_dependent:
            return NoiseCompilation(
                model=None,
                technology_id=self.technology_id,
                technology_version=stim.__version__,
                source_spec_digest=digest,
                support=NoiseApproximation.UNSUPPORTED,
                reason=(
                    f"stim-noise adapter only compiles time-independent 'independent-phase-flip'; "
                    f"requested family={spec.family!r}, time_dependent={spec.time_dependent}"
                ),
            )
        noise = build_independent_phase_flip(spec=spec)
        return NoiseCompilation(
            model=StimPauliChannel("Z_ERROR", noise.physical_error_rate),
            technology_id=self.technology_id,
            technology_version=stim.__version__,
            source_spec_digest=digest,
            support=NoiseApproximation.EXACT,
        )


@NOISE_ADAPTERS.register(TechnologyId.STIM_NOISE.value)
def build_stim_noise_compiler() -> StimNoiseCompiler:
    return StimNoiseCompiler()
