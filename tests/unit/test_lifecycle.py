import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import ai_qec.notebook_api as qec
from ai_qec.experiment.local import LocalArtifactRepository, LocalExperiment
from ai_qec.experiment.streams import DerivedRandomStreams
from tests.helpers import requires_numpy


class RandomStreamTests(unittest.TestCase):
    def test_descriptors_are_stable_and_distinct(self) -> None:
        streams = DerivedRandomStreams(2017)
        first = streams.describe("training_shuffle")
        self.assertEqual(first, DerivedRandomStreams(2017).describe("training_shuffle"))
        self.assertNotEqual(first.derived_seed, streams.describe("qec_sampling").derived_seed)
        self.assertNotEqual(
            first.derived_seed, DerivedRandomStreams(2018).describe("training_shuffle").derived_seed
        )
        self.assertLess(first.derived_seed, 2**63)

    @requires_numpy
    def test_consuming_one_stream_leaves_others_unchanged(self) -> None:
        reference = DerivedRandomStreams(7).numpy("dataset_split").random(5).tolist()
        streams = DerivedRandomStreams(7)
        streams.numpy("training_shuffle").random(1000)
        self.assertEqual(streams.numpy("dataset_split").random(5).tolist(), reference)
        snapshot = streams.snapshot("dataset_split")
        expected = streams.numpy("dataset_split").random(3).tolist()
        streams.restore("dataset_split", snapshot)
        self.assertEqual(streams.numpy("dataset_split").random(3).tolist(), expected)


class AttemptLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.experiment = LocalExperiment(self.root / "runs", "exp-1", spec=None)

    def test_terminal_attempts_are_immutable(self) -> None:
        attempt = self.experiment.start_attempt()
        self.assertIs(attempt.status, qec.AttemptStatus.RUNNING)
        attempt.fail("boom")
        for transition in (
            attempt.complete,
            lambda: attempt.interrupt("x"),
            lambda: attempt.fail("again"),
        ):
            with self.assertRaises(qec.AttemptStateError):
                transition()
        self.assertIs(attempt.status, qec.AttemptStatus.FAILED)
        self.assertEqual(attempt.record["reason"], "boom")

    def test_recovery_attempt_references_its_terminal_source(self) -> None:
        first = self.experiment.start_attempt()
        plan = qec.RecoveryPlan(source=qec.RecoverySource("exp-1", first.attempt_id, "running"))
        with self.assertRaises(qec.AttemptStateError):
            self.experiment.recover(plan)
        first.interrupt("kernel restart")
        second = self.experiment.recover(
            qec.RecoveryPlan(
                source=qec.RecoverySource("exp-1", first.attempt_id, "interrupted"),
                resume_from_stage="training",
            )
        )
        self.assertEqual(second.attempt_id, "attempt-0002")
        self.assertEqual(
            second.recovery_from, {"attempt_id": "attempt-0001", "terminal_status": "interrupted"}
        )
        self.assertIs(first.status, qec.AttemptStatus.INTERRUPTED)

    def test_owner_liveness(self) -> None:
        attempt = self.experiment.start_attempt()
        self.assertFalse(attempt.owner_is_other_live_process())
        finished = subprocess.run(
            [sys.executable, "-c", "import os; print(os.getpid())"],
            capture_output=True,
            text=True,
            check=True,
        )
        record = attempt.record
        record["owner"]["pid"] = int(finished.stdout)
        attempt._write(record)
        self.assertFalse(attempt.owner_is_other_live_process())
        record["owner"]["pid"] = os.getppid()
        attempt._write(record)
        self.assertTrue(attempt.owner_is_other_live_process())


class ArtifactRepositoryTests(unittest.TestCase):
    def test_commit_verify_and_detect_modification(self) -> None:
        root = Path(tempfile.mkdtemp())
        repository = LocalArtifactRepository(root, root / "runs" / "exp-1")
        path = root / "runs" / "exp-1" / "attempts" / "attempt-0001" / "metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"x": 1}\n', encoding="utf-8")
        ref = repository.commit_file(
            path,
            kind=qec.ArtifactKind.METRICS,
            name="metrics",
            attempt_id="attempt-0001",
            stage_id="scientific_evaluation",
            media_type="application/json",
        )
        self.assertEqual(ref.uri, "runs/exp-1/attempts/attempt-0001/metrics.json")
        self.assertTrue(repository.verify(ref))
        self.assertEqual(repository.read_json(ref.artifact_id), {"x": 1})
        self.assertEqual(
            repository.manifest_for(ref.artifact_id).producer_stage_id, "scientific_evaluation"
        )
        path.write_text('{"x": 2}\n', encoding="utf-8")
        self.assertFalse(repository.verify(ref))
        with self.assertRaises(qec.AttemptStateError):
            repository.read_json(ref.artifact_id)


if __name__ == "__main__":
    unittest.main()
