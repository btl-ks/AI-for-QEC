"""Auditable scientific metric and gate result values."""

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    """Point estimate with explicit numerator, denominator, and interval."""

    name: str
    value: float
    numerator: int
    denominator: int
    confidence_level: float
    interval_low: float
    interval_high: float
    method: str


@dataclass(frozen=True, slots=True)
class DecoderEvaluation:
    """Scientific result for one decoder over one fixed shot set."""

    decoder_id: str
    dataset_artifact_id: str
    sample_ids_digest: str
    logical_error_rate: MetricEstimate
    failure_count: int
    timeout_count: int
    not_converged_count: int
    metrics_artifact_id: str


@dataclass(frozen=True, slots=True)
class ScientificEvaluationResult:
    """Comparable AI and baseline results produced under one protocol."""

    evaluation_id: str
    spec_digest: str
    dataset_artifact_id: str
    decoder_results: tuple[DecoderEvaluation, ...]
    result_artifact_id: str


class GateDecision(StrEnum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class ScientificAcceptanceResult:
    """Persistable and independently reviewable Accuracy Gate outcome."""

    gate_id: str
    decision: GateDecision
    evaluation_id: str
    gate_spec_digest: str
    evidence_artifact_ids: tuple[str, ...]
    rationale: str
