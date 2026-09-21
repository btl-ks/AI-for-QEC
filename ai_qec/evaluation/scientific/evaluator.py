"""Scientific evaluator protocol; no metrics are computed here."""

from typing import Mapping, Protocol, TypeVar, runtime_checkable

from ai_qec.data.datasets.artifact import DatasetArtifact
from ai_qec.models.decoders.protocol import Decoder

from .result import ScientificEvaluationResult
from .spec import ScientificEvaluationSpec


ArrayT = TypeVar("ArrayT")


@runtime_checkable
class ScientificEvaluator(Protocol[ArrayT]):
    """Evaluate all declared decoders on one immutable test artifact."""

    def evaluate(
        self,
        spec: ScientificEvaluationSpec,
        dataset: DatasetArtifact,
        decoders: Mapping[str, Decoder[ArrayT]],
        attempt_id: str,
    ) -> ScientificEvaluationResult: ...
