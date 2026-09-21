"""Quantum error-correction problem specification."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QECSpec:
    """Stable identity inputs for a QEC problem.

    Validation and backend compilation are deliberately outside this value object.
    """

    code_family: str
    distance: int
    rounds: int
    logical_basis: str
    circuit_family: str
    schema_version: str = "qec-spec-v1"
