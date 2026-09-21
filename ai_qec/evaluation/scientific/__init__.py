"""Scientific evaluation and Accuracy Gate contracts."""

from .accuracy_gate import AccuracyGate
from .evaluator import ScientificEvaluator
from .result import (
    DecoderEvaluation,
    GateDecision,
    MetricEstimate,
    ScientificAcceptanceResult,
    ScientificEvaluationResult,
)
from .spec import AccuracyGateSpec, ScientificEvaluationSpec

__all__ = [
    "AccuracyGate",
    "AccuracyGateSpec",
    "DecoderEvaluation",
    "GateDecision",
    "MetricEstimate",
    "ScientificAcceptanceResult",
    "ScientificEvaluationResult",
    "ScientificEvaluationSpec",
    "ScientificEvaluator",
]
