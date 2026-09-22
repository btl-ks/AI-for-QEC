"""Local scientific evaluator and paired (absolute or relative) non-inferiority Accuracy Gate.

Every decoder receives the same host QECBatch of the test split. A shot fails
when any predicted observable flip differs from the truth; invalid shots
(timeouts, non-convergence) are handled by the pre-registered
``invalid_sample_policy``. The evaluator persists predictions, per-decoder
metrics, and the paired failure tables the Gate needs as its evidence.
"""

from collections.abc import Callable, Mapping
import hashlib
from itertools import product
from math import isfinite
from pathlib import Path
import time

from ai_qec.data.datasets.artifact import DatasetArtifact
from ai_qec.data.schema.batch import QECBatch
from ai_qec.experiment.artifact import ArtifactKind, ArtifactRef
from ai_qec.models.decoders.protocol import DecodeRequest, DecodeStatus
from ai_qec.utils.hashing import sha256_json, to_jsonable, write_json_atomic

from .result import (
    DecoderEvaluation,
    GateDecision,
    MetricEstimate,
    ScientificAcceptanceResult,
    ScientificEvaluationResult,
)
from .spec import AccuracyGateSpec, ScientificEvaluationSpec
from .statistics import paired_difference_interval, wilson_interval

INVALID_SAMPLE_POLICIES = ("count-as-failure", "fail")
ABSOLUTE_RULE = "paired-non-inferiority"
RELATIVE_RULE = "paired-relative-non-inferiority"
GATE_RULES = (ABSOLUTE_RULE, RELATIVE_RULE)
PRIMARY_METRIC = "logical_error_rate"


class EvaluationError(RuntimeError):
    """A decoder result cannot be scored under the declared protocol."""


class AccuracyGateError(RuntimeError):
    """The Gate rule or its evidence is not usable; no decision is produced."""


def sample_ids_digest(sample_ids) -> str:
    return "sha256:" + hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def class_labels(num_observables: int) -> tuple[str, ...]:
    """Residual logical classes as observable-flip bit strings; bit i is observable i."""

    return tuple("".join(bits) for bits in product("01", repeat=num_observables))


class LocalScientificEvaluator:
    def __init__(
        self,
        *,
        load_test_batch: Callable[[DatasetArtifact], QECBatch],
        candidate_decoder_id: str,
        protocol: Mapping[str, object],
        output_dir: Path,
        commit_artifact: Callable[..., ArtifactRef],
    ) -> None:
        self.load_test_batch = load_test_batch
        self.candidate_decoder_id = candidate_decoder_id
        self.protocol = dict(protocol)
        self.output_dir = Path(output_dir)
        self.commit_artifact = commit_artifact

    def evaluate(
        self,
        spec: ScientificEvaluationSpec,
        dataset: DatasetArtifact,
        decoders: Mapping[str, object],
        attempt_id: str,
    ) -> ScientificEvaluationResult:
        import numpy as np

        if spec.primary_metric != PRIMARY_METRIC:
            raise EvaluationError(f"unsupported primary metric {spec.primary_metric!r}")
        if spec.invalid_sample_policy not in INVALID_SAMPLE_POLICIES:
            raise EvaluationError(
                f"unsupported invalid_sample_policy {spec.invalid_sample_policy!r}"
            )
        if self.candidate_decoder_id not in decoders:
            raise EvaluationError(
                f"candidate decoder {self.candidate_decoder_id!r} was not supplied"
            )

        batch = self.load_test_batch(dataset)
        truth = np.asarray(batch.observable_truth, dtype=np.int8)
        digest = sample_ids_digest(batch.sample_ids)
        index_of = {sample_id: index for index, sample_id in enumerate(batch.sample_ids)}
        labels = class_labels(truth.shape[1])
        spec_digest = sha256_json(
            {"spec": spec, "candidate": self.candidate_decoder_id, "protocol": self.protocol}
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        evaluations: dict[str, DecoderEvaluation] = {}
        failures: dict[str, object] = {}
        details: dict[str, object] = {}
        order = [
            self.candidate_decoder_id,
            *(name for name in decoders if name != self.candidate_decoder_id),
        ]
        for decoder_id in order:
            decoder = decoders[decoder_id]
            request = DecodeRequest(
                request_id=f"{attempt_id}.{decoder_id}.test",
                decoder_id=decoder_id,
                batch=batch,
                context={"split": "test", "dataset_artifact_id": dataset.artifact_id},
            )
            started = time.perf_counter()
            result = decoder.decode(request)
            seconds = time.perf_counter() - started
            if result.decoder_id != decoder_id or result.request_id != request.request_id:
                raise EvaluationError(f"{decoder_id} returned a result for a different request")
            if (
                result.status in (DecodeStatus.UNSUPPORTED, DecodeStatus.FAILED)
                or result.predictions is None
            ):
                raise EvaluationError(
                    f"{decoder_id} could not decode the test split: {result.error}"
                )
            predictions = np.asarray(result.predictions, dtype=np.int8)
            if predictions.shape != truth.shape:
                raise EvaluationError(
                    f"{decoder_id} predictions have shape {predictions.shape}, expected {truth.shape}"
                )
            invalid = np.zeros(len(truth), dtype=bool)
            unknown = [
                sample_id for sample_id in result.failed_sample_ids if sample_id not in index_of
            ]
            if unknown:
                raise EvaluationError(
                    f"{decoder_id} reported unknown failed sample ids, e.g. {unknown[0]}"
                )
            invalid[[index_of[sample_id] for sample_id in result.failed_sample_ids]] = True
            if np.any(~np.isin(predictions[~invalid], (0, 1))) or np.any(
                predictions[invalid] != -1
            ):
                raise EvaluationError(f"{decoder_id} mixes invalid markers and predictions")
            if invalid.any() and spec.invalid_sample_policy == "fail":
                raise EvaluationError(
                    f"{decoder_id} produced {int(invalid.sum())} invalid samples and invalid_sample_policy is 'fail'"
                )
            failed = invalid | np.any(predictions != truth, axis=1)
            residual = (predictions[~invalid] ^ truth[~invalid]).astype(np.uint8)
            residual_labels = ["".join(str(bit) for bit in row) for row in residual]
            counts = {label: residual_labels.count(label) for label in labels}
            numerator, denominator = int(failed.sum()), len(failed)
            low, high = wilson_interval(numerator, denominator, spec.confidence_level)
            estimate = MetricEstimate(
                name=PRIMARY_METRIC,
                value=numerator / denominator,
                numerator=numerator,
                denominator=denominator,
                confidence_level=spec.confidence_level,
                interval_low=low,
                interval_high=high,
                method="wilson",
            )
            timeouts = int(invalid.sum()) if result.status is DecodeStatus.TIMED_OUT else 0
            not_converged = int(invalid.sum()) if result.status is DecodeStatus.NOT_CONVERGED else 0

            predictions_path = self.output_dir / f"{decoder_id}.predictions.npz"
            np.savez(predictions_path, predictions=predictions, failed=failed, invalid=invalid)
            predictions_ref = self.commit_artifact(
                predictions_path,
                kind=ArtifactKind.PREDICTIONS,
                name=f"{decoder_id}.predictions",
                media_type="application/x-npz",
                metadata={"decoder_id": decoder_id, "sample_ids_digest": digest},
            )
            metrics = {
                "decoder_id": decoder_id,
                "dataset_artifact_id": dataset.artifact_id,
                "sample_ids_digest": digest,
                "status": result.status.value,
                "logical_error_rate": to_jsonable(estimate),
                "timeout_count": timeouts,
                "not_converged_count": not_converged,
                "invalid_sample_policy": spec.invalid_sample_policy,
                "logical_class_counts": counts,
                "runtime": to_jsonable(result.runtime),
                "provenance": to_jsonable(result.provenance),
                "decode_seconds": seconds,
                "predictions_artifact_id": predictions_ref.artifact_id,
            }
            metrics_path = self.output_dir / f"{decoder_id}.metrics.json"
            write_json_atomic(metrics_path, metrics)
            metrics_ref = self.commit_artifact(
                metrics_path,
                kind=ArtifactKind.METRICS,
                name=f"{decoder_id}.metrics",
                media_type="application/json",
                metadata={"decoder_id": decoder_id},
            )
            evaluations[decoder_id] = DecoderEvaluation(
                decoder_id=decoder_id,
                dataset_artifact_id=dataset.artifact_id,
                sample_ids_digest=digest,
                logical_error_rate=estimate,
                failure_count=numerator,
                timeout_count=timeouts,
                not_converged_count=not_converged,
                metrics_artifact_id=metrics_ref.artifact_id,
                logical_class_counts=counts,
            )
            failures[decoder_id] = failed
            details[decoder_id] = metrics

        candidate = failures[self.candidate_decoder_id]
        paired = {}
        for decoder_id in order[1:]:
            baseline = failures[decoder_id]
            paired[decoder_id] = {
                "both_fail": int(np.sum(candidate & baseline)),
                "candidate_only_fail": int(np.sum(candidate & ~baseline)),
                "baseline_only_fail": int(np.sum(~candidate & baseline)),
                "neither_fail": int(np.sum(~candidate & ~baseline)),
            }
        evaluation_id = f"{attempt_id}.scientific-evaluation"
        evaluation_path = self.output_dir / "scientific_evaluation.json"
        write_json_atomic(
            evaluation_path,
            {
                "evaluation_id": evaluation_id,
                "spec": to_jsonable(spec),
                "protocol": self.protocol,
                "spec_digest": spec_digest,
                "dataset_artifact_id": dataset.artifact_id,
                "sample_ids_digest": digest,
                "candidate": self.candidate_decoder_id,
                "baselines": order[1:],
                "decoders": details,
                "paired": paired,
            },
        )
        evaluation_ref = self.commit_artifact(
            evaluation_path,
            kind=ArtifactKind.METRICS,
            name="scientific-evaluation",
            media_type="application/json",
            metadata={"evaluation_id": evaluation_id},
        )
        return ScientificEvaluationResult(
            evaluation_id=evaluation_id,
            spec_digest=spec_digest,
            dataset_artifact_id=dataset.artifact_id,
            decoder_results=tuple(evaluations[decoder_id] for decoder_id in order),
            result_artifact_id=evaluation_ref.artifact_id,
        )


class LocalAccuracyGate:
    """Applies only the pre-registered rule to the persisted paired evidence."""

    def __init__(
        self,
        *,
        read_artifact_json: Callable[[str], Mapping[str, object]],
        output_dir: Path,
        commit_artifact: Callable[..., ArtifactRef],
    ) -> None:
        self.read_artifact_json = read_artifact_json
        self.output_dir = Path(output_dir)
        self.commit_artifact = commit_artifact

    def assess(
        self,
        spec: AccuracyGateSpec,
        result: ScientificEvaluationResult,
        attempt_id: str,
    ) -> ScientificAcceptanceResult:
        if spec.comparison_rule not in GATE_RULES:
            raise AccuracyGateError(
                f"unsupported comparison_rule {spec.comparison_rule!r}; supported: {GATE_RULES}"
            )
        if spec.primary_metric != PRIMARY_METRIC:
            raise AccuracyGateError(f"unsupported gate metric {spec.primary_metric!r}")
        evidence = self.read_artifact_json(result.result_artifact_id)
        if evidence.get("evaluation_id") != result.evaluation_id:
            raise AccuracyGateError(
                "evaluation evidence does not belong to this ScientificEvaluationResult"
            )
        candidate = str(evidence["candidate"])
        tables = evidence["paired"]
        if spec.baseline_decoder not in tables:
            raise AccuracyGateError(
                f"baseline {spec.baseline_decoder!r} was not evaluated; evaluated: {sorted(tables)}"
            )
        table = tables[spec.baseline_decoder]
        counts = (
            table["both_fail"],
            table["candidate_only_fail"],
            table["baseline_only_fail"],
            table["neither_fail"],
        )
        candidate_rate = (counts[0] + counts[1]) / sum(counts)
        baseline_rate = (counts[0] + counts[2]) / sum(counts)
        ratio = None if baseline_rate == 0 else candidate_rate / baseline_rate
        baseline = spec.baseline_decoder
        if spec.comparison_rule == RELATIVE_RULE:
            if not (isfinite(spec.tolerance) and spec.tolerance >= 0.0):
                raise AccuracyGateError(
                    f"{RELATIVE_RULE} needs a non-negative relative tolerance, got {spec.tolerance!r}"
                )
            scale, threshold = 1.0 + spec.tolerance, 0.0
            method = "mover-wilson-linear-contrast"
            rule_parameters = {"relative_tolerance": spec.tolerance, "scale": scale}
        else:
            scale, threshold = 1.0, spec.tolerance
            method = "newcombe-1998-method-10"
            rule_parameters = {"absolute_tolerance": spec.tolerance}
        difference, low, high = paired_difference_interval(
            *counts, spec.confidence_level, scale=scale
        )
        decision = GateDecision.PASS if high <= threshold else GateDecision.FAIL
        if spec.comparison_rule == RELATIVE_RULE:
            ratio_text = "undefined" if ratio is None else f"{ratio:.3f}"
            rationale = (
                f"{spec.comparison_rule}: LER({candidate}) / LER({baseline}) = {ratio_text}; "
                f"LER({candidate}) - {scale:g} x LER({baseline}) = {difference:+.4f}; "
                f"{spec.confidence_level:.0%} MOVER paired interval [{low:+.4f}, {high:+.4f}]; "
                f"PASS requires upper bound <= 0 (relative tolerance {spec.tolerance * 100:g}%) "
                f"-> {decision.value.upper()}"
            )
        else:
            rationale = (
                f"{spec.comparison_rule}: LER({candidate}) - LER({baseline}) = {difference:+.4f}; "
                f"{spec.confidence_level:.0%} Newcombe paired interval [{low:+.4f}, {high:+.4f}]; "
                f"PASS requires upper bound <= tolerance {spec.tolerance:+.4f} -> {decision.value.upper()}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "acceptance.json"
        write_json_atomic(
            path,
            {
                "gate_spec": to_jsonable(spec),
                "evaluation_id": result.evaluation_id,
                "candidate": candidate,
                "baseline": spec.baseline_decoder,
                "paired_table": table,
                "rule_parameters": rule_parameters,
                "ratio_estimate": ratio,
                "difference": difference,
                "interval": [low, high],
                "method": method,
                "decision": decision.value,
                "rationale": rationale,
            },
        )
        ref = self.commit_artifact(
            path,
            kind=ArtifactKind.ACCURACY_GATE,
            name="accuracy-gate",
            media_type="application/json",
            metadata={"gate_id": spec.gate_id, "decision": decision.value},
        )
        return ScientificAcceptanceResult(
            gate_id=spec.gate_id,
            decision=decision,
            evaluation_id=result.evaluation_id,
            gate_spec_digest=sha256_json(spec),
            evidence_artifact_ids=(result.result_artifact_id, ref.artifact_id),
            rationale=rationale,
        )
