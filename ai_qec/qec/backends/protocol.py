"""Backend compatibility and compilation protocols."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ai_qec.qec.noise import NoiseApproximation, NoiseSpec
from ai_qec.qec.spec import QECSpec


@dataclass(frozen=True, slots=True)
class BackendCompatibility:
    """A backend's explicit response to requested QEC/noise semantics."""

    backend_id: str
    support: NoiseApproximation
    reason: str | None = None
    approximation_id: str | None = None


@runtime_checkable
class QECBackend(Protocol):
    """Contract for a concrete simulator or hardware adapter."""

    @property
    def backend_id(self) -> str: ...

    def check_compatibility(
        self,
        qec: QECSpec,
        noise: NoiseSpec,
    ) -> BackendCompatibility: ...
