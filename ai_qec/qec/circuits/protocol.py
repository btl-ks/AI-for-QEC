"""Vendor-neutral circuit construction boundary."""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from ai_qec.qec.spec import QECSpec


CircuitT = TypeVar("CircuitT")


@dataclass(frozen=True, slots=True)
class CircuitBuildResult(Generic[CircuitT]):
    """A third-party circuit object with auditable adapter provenance."""

    circuit: CircuitT
    technology_id: str
    technology_version: str
    qec_spec_digest: str
    circuit_digest: str


@runtime_checkable
class QECCircuitAdapter(Protocol[CircuitT]):
    """Build a circuit without exposing its library in QECSpec."""

    @property
    def technology_id(self) -> str: ...

    def build(self, spec: QECSpec) -> CircuitBuildResult[CircuitT]: ...

