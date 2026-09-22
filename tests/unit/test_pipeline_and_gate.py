import json
from pathlib import Path
import tempfile
import unittest

import ai_qec.notebook_api as qec
from ai_qec.evaluation.scientific.local import LocalAccuracyGate
from tests.helpers import requires_runtime


def _gate_fixture(table: dict[str, int]):
    root = Path(tempfile.mkdtemp())
    committed = []

    def commit(path, *, kind, name, media_type, metadata=None):
        committed.append(Path(path))
        return qec.ArtifactRef(f"attempt-0001.{name}", kind, str(path), "sha256:x", media_type)

    evidence = {
        "evaluation_id": "attempt-0001.scientific-evaluation",
        "candidate": "ai",
        "paired": {"mwpm": table},
    }
    gate = LocalAccuracyGate(
        read_artifact_json=lambda _: evidence, output_dir=root, commit_artifact=commit
    )
    estimate = qec.MetricEstimate("logical_error_rate", 0.1, 1, 10, 0.95, 0.0, 0.4, "wilson")
    result = qec.ScientificEvaluationResult(
        evaluation_id="attempt-0001.scientific-evaluation",
        spec_digest="sha256:spec",
        dataset_artifact_id="ds-1",
        decoder_results=(
            qec.DecoderEvaluation("ai", "ds-1", "sha256:ids", estimate, 1, 0, 0, "m1"),
        ),
        result_artifact_id="attempt-0001.scientific-evaluation",
    )
    return gate, result, committed


def _gate_spec(rule: str = "paired-non-inferiority", tolerance: float = 0.02):
    return qec.AccuracyGateSpec("gate-a", "mwpm", "logical_error_rate", rule, tolerance, 0.95)


class AccuracyGateTests(unittest.TestCase):
    def test_pass_and_fail_follow_the_upper_bound(self) -> None:
        close = {
            "both_fail": 900,
            "candidate_only_fail": 110,
            "baseline_only_fail": 100,
            "neither_fail": 8890,
        }
        gate, result, committed = _gate_fixture(close)
        acceptance = gate.assess(_gate_spec(), result, "attempt-0001")
        self.assertIs(acceptance.decision, qec.GateDecision.PASS)
        self.assertEqual(acceptance.evidence_artifact_ids[0], result.result_artifact_id)
        self.assertIn("tolerance +0.0200", acceptance.rationale)
        self.assertTrue(committed[0].is_file())

        worse = {
            "both_fail": 900,
            "candidate_only_fail": 600,
            "baseline_only_fail": 100,
            "neither_fail": 8400,
        }
        gate, result, _ = _gate_fixture(worse)
        self.assertIs(
            gate.assess(_gate_spec(), result, "attempt-0001").decision, qec.GateDecision.FAIL
        )

    def test_relative_rule_scales_the_margin_with_the_baseline(self) -> None:
        low_p = {
            "both_fail": 247,
            "candidate_only_fail": 184,
            "baseline_only_fail": 137,
            "neither_fail": 9432,
        }
        gate, result, committed = _gate_fixture(low_p)
        self.assertIs(
            gate.assess(_gate_spec(), result, "attempt-0001").decision, qec.GateDecision.PASS
        )
        relative = _gate_spec("paired-relative-non-inferiority", 0.15)
        acceptance = gate.assess(relative, result, "attempt-0001")
        self.assertIs(acceptance.decision, qec.GateDecision.FAIL)
        self.assertIn("LER(ai) / LER(mwpm) = 1.122", acceptance.rationale)
        self.assertIn("relative tolerance 15%", acceptance.rationale)
        evidence = json.loads(committed[-1].read_text())
        self.assertEqual(evidence["rule_parameters"], {"relative_tolerance": 0.15, "scale": 1.15})
        self.assertAlmostEqual(evidence["ratio_estimate"], 431 / 384)
        self.assertGreater(evidence["interval"][1], 0.0)
        self.assertEqual(evidence["method"], "mover-wilson-linear-contrast")

        high_p = {
            "both_fail": 4205,
            "candidate_only_fail": 1555,
            "baseline_only_fail": 1230,
            "neither_fail": 3010,
        }
        gate, result, _ = _gate_fixture(high_p)
        self.assertIs(
            gate.assess(_gate_spec(), result, "attempt-0001").decision, qec.GateDecision.FAIL
        )
        self.assertIs(
            gate.assess(relative, result, "attempt-0001").decision, qec.GateDecision.PASS
        )

    def test_negative_relative_tolerance_produces_no_decision(self) -> None:
        table = {
            "both_fail": 1,
            "candidate_only_fail": 1,
            "baseline_only_fail": 1,
            "neither_fail": 1,
        }
        gate, result, committed = _gate_fixture(table)
        with self.assertRaises(qec.AccuracyGateError):
            gate.assess(
                _gate_spec("paired-relative-non-inferiority", -0.1), result, "attempt-0001"
            )
        self.assertEqual(committed, [])

    def test_unknown_rule_or_baseline_produces_no_decision(self) -> None:
        table = {
            "both_fail": 1,
            "candidate_only_fail": 1,
            "baseline_only_fail": 1,
            "neither_fail": 1,
        }
        gate, result, committed = _gate_fixture(table)
        with self.assertRaises(qec.AccuracyGateError):
            gate.assess(_gate_spec(rule="superiority"), result, "attempt-0001")
        other = qec.AccuracyGateSpec(
            "gate-a", "other", "logical_error_rate", "paired-non-inferiority", 0.0, 0.95
        )
        with self.assertRaises(qec.AccuracyGateError):
            gate.assess(other, result, "attempt-0001")
        self.assertEqual(committed, [])


@requires_runtime
class HostToDevicePipelineTests(unittest.TestCase):
    def _batch(self):
        import numpy as np

        events = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
        return qec.QECBatch(
            detector_events=events,
            observable_truth=np.array([[0, 1], [1, 1]], dtype=np.uint8),
            physical_errors=np.array([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.uint8),
            sample_ids=("test-0000000", "test-0000001"),
            dataset_artifact_id="ds-1",
            layout=qec.BatchLayout(
                qec.BatchRepresentation.NUMPY_ARRAY,
                qec.MemoryResidency.HOST,
                "cpu",
                "uint8",
                (2, 3),
            ),
            context={"split": "test"},
        )

    def _check(
        self, device: str, residency: qec.MemoryResidency, representation: qec.BatchRepresentation
    ) -> None:
        import numpy as np

        from ai_qec.data.loaders.pytorch import build_torch_h2d_pipeline

        batch = self._batch()
        pipeline = build_torch_h2d_pipeline(spec=qec.CPUToGPUPipelineSpec(), device=device)
        moved, evidence = pipeline.transfer(batch)
        self.assertEqual(
            (moved.sample_ids, moved.dataset_artifact_id, moved.context),
            (batch.sample_ids, "ds-1", {"split": "test"}),
        )
        self.assertEqual(str(moved.detector_events.device), moved.layout.device)
        self.assertEqual(
            (moved.layout.residency, moved.layout.representation), (residency, representation)
        )
        self.assertEqual(evidence.source_layout, batch.layout)
        self.assertEqual(evidence.target_layout, moved.layout)
        self.assertFalse(evidence.host_staging_observed)
        self.assertTrue(np.array_equal(moved.physical_errors.cpu().numpy(), batch.physical_errors))

    def test_cpu_transfer_preserves_identity(self) -> None:
        self._check("cpu", qec.MemoryResidency.HOST, qec.BatchRepresentation.PYTORCH_TENSOR)

    def test_cuda_transfer_records_device_layout(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        self._check(
            "cuda:0", qec.MemoryResidency.CUDA_DEVICE, qec.BatchRepresentation.PYTORCH_CUDA_TENSOR
        )


if __name__ == "__main__":
    unittest.main()
