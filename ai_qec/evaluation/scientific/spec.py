"""Pre-registered scientific evaluation and acceptance specifications."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScientificEvaluationSpec:
    """Fair-comparison protocol shared by AI and baseline decoders."""

    baseline_decoders: tuple[str, ...]
    primary_metric: str
    confidence_level: float
    stopping_rule: str
    invalid_sample_policy: str
    noise_points: tuple[float, ...] = ()
    code_distances: tuple[int, ...] = ()
    schema_version: str = "scientific-evaluation-spec-v1"


@dataclass(frozen=True, slots=True)
class AccuracyGateSpec:
    """Pre-registered statistical decision rule for Gate A or Gate B."""

    gate_id: str
    baseline_decoder: str
    primary_metric: str
    comparison_rule: str
    tolerance: float
    confidence_level: float
    stage: str = "gate-a"
    schema_version: str = "accuracy-gate-spec-v1"
