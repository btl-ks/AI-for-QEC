"""Accuracy Gate decision protocol; no statistical rule is implemented."""

from typing import Protocol, runtime_checkable

from .result import ScientificAcceptanceResult, ScientificEvaluationResult
from .spec import AccuracyGateSpec


@runtime_checkable
class AccuracyGate(Protocol):
    """Apply only a pre-registered gate rule to complete scientific evidence."""

    def assess(
        self,
        spec: AccuracyGateSpec,
        result: ScientificEvaluationResult,
        attempt_id: str,
    ) -> ScientificAcceptanceResult: ...
