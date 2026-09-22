from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.helpers import requires_runtime, run_workflow, tiny_config


def _stage_statuses(run) -> dict[str, str]:
    from ai_qec.paper.local_runtime import STAGE_ORDER

    statuses = {}
    for name in STAGE_ORDER:
        record = run._attempt.stages.get(run.attempt_id, name)
        statuses[name] = None if record is None else record.status.value
    return statuses


def _stage_metadata(run, name: str) -> dict:
    import json

    return json.loads((run.directory / "stages" / f"{name}.meta.json").read_text(encoding="utf-8"))


@requires_runtime
class LocalRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        import ai_qec.notebook_api as qec

        self.qec = qec
        self.root = Path(tempfile.mkdtemp())
        self.runtime = qec.LocalNotebookPlatform(self.root, verbose=False)

    def test_end_to_end_then_second_run_reuses_verified_stages(self) -> None:
        qec = self.qec
        run, model, scientific, acceptance, performance, figures = run_workflow(
            self.runtime, tiny_config()
        )
        self.assertIs(run.status, qec.AttemptStatus.COMPLETED)
        self.assertTrue(all(status == "completed" for status in _stage_statuses(run).values()))
        self.assertEqual(
            [item.decoder_id for item in scientific.decoder_results],
            ["joint-error-syndrome-rbm", "pymatching-cpu-decoder"],
        )
        for item in scientific.decoder_results:
            self.assertEqual(item.logical_error_rate.denominator, 200)
            self.assertEqual(sum(item.logical_class_counts.values()) + item.timeout_count, 200)
            self.assertEqual(set(item.logical_class_counts), {"00", "01", "10", "11"})
        self.assertEqual(len({item.sample_ids_digest for item in scientific.decoder_results}), 1)
        self.assertIs(acceptance.decision, qec.GateDecision.PASS)
        self.assertIsNotNone(performance)
        self.assertEqual(len(figures), 3)
        self.assertTrue(all(self.runtime.artifact_path(ref).is_file() for ref in figures))

        again = run_workflow(self.runtime, tiny_config())
        second = again[0]
        self.assertEqual(second.attempt_id, "attempt-0002")
        self.assertEqual(
            second._attempt.recovery_from,
            {"attempt_id": "attempt-0001", "terminal_status": "completed"},
        )
        self.assertTrue(_stage_metadata(second, "dataset")["cache_hit"])
        self.assertEqual(_stage_metadata(second, "training")["reused_from"], "attempt-0001")
        self.assertEqual(
            _stage_metadata(second, "scientific_evaluation")["reused_from"], "attempt-0001"
        )
        self.assertEqual(again[1], model)
        self.assertEqual(again[2], scientific)

    def test_interrupted_training_resumes_and_matches_uninterrupted_training(self) -> None:
        import torch

        from ai_qec.training.trainers.pytorch_trainer import PyTorchTrainer

        config = tiny_config(**{"training.epochs": 4})
        reference_runtime = self.qec.LocalNotebookPlatform(Path(tempfile.mkdtemp()), verbose=False)
        _, reference_model, *_ = run_workflow(reference_runtime, config)

        original = PyTorchTrainer.after_epoch

        def interrupt_after_second_epoch(trainer, epoch, record):
            original(trainer, epoch, record)
            if epoch == 2:
                raise KeyboardInterrupt

        with (
            mock.patch.object(PyTorchTrainer, "after_epoch", interrupt_after_second_epoch),
            self.assertRaises(KeyboardInterrupt),
        ):
            run_workflow(self.runtime, config)
        experiment = self.runtime.create_experiment(config)
        first = experiment.experiment.attempt("attempt-0001")
        self.assertIs(first.status, self.qec.AttemptStatus.INTERRUPTED)
        self.assertEqual(first.stages.get("attempt-0001", "training").status.value, "interrupted")

        run, model, *_ = run_workflow(self.runtime, config)
        self.assertEqual(run.attempt_id, "attempt-0002")
        training = _stage_metadata(run, "training")
        self.assertEqual(training["resumed_from"]["epoch"], 2)
        self.assertEqual(training["resumed_from"]["source_attempt_id"], "attempt-0001")
        self.assertEqual(training["epochs_trained_in_attempt"], 2)
        self.assertIs(first.status, self.qec.AttemptStatus.INTERRUPTED)

        resumed = torch.load(self.runtime.artifact_path(_checkpoint(model)), weights_only=True)[
            "state_dict"
        ]
        straight = torch.load(
            reference_runtime.artifact_path(_checkpoint(reference_model)), weights_only=True
        )["state_dict"]
        for name in straight:
            self.assertTrue(torch.equal(resumed[name], straight[name]), name)
        self.assertEqual(model.model_identity, reference_model.model_identity)

    def test_modified_model_checkpoint_is_not_reused(self) -> None:
        _, model, *_ = run_workflow(self.runtime, tiny_config())
        path = self.runtime.artifact_path(_checkpoint(model))
        path.write_bytes(path.read_bytes() + b"tampered")
        run, second_model, *_ = run_workflow(self.runtime, tiny_config())
        training = _stage_metadata(run, "training")
        self.assertNotIn("reused_from", training)
        self.assertEqual(training["resumed_from"]["epoch"], 3)
        self.assertEqual(training["epochs_trained_in_attempt"], 0)
        self.assertEqual(second_model.artifact_id, "attempt-0002.model-final")
        self.assertEqual(second_model.model_identity, model.model_identity)

    def test_gate_only_change_reuses_stages_across_experiments(self) -> None:
        qec = self.qec
        first_run, first_model, first_scientific, *_ = run_workflow(self.runtime, tiny_config())
        relative = tiny_config(
            **{
                "accuracy_gate.comparison_rule": "paired-relative-non-inferiority",
                "accuracy_gate.tolerance": 100.0,
            }
        )
        run, model, scientific, acceptance, performance, figures = run_workflow(
            self.runtime, relative
        )
        self.assertNotEqual(run.experiment_id, first_run.experiment_id)
        self.assertEqual(run.attempt_id, "attempt-0001")
        self.assertIs(run.status, qec.AttemptStatus.COMPLETED)
        for stage in ("training", "scientific_evaluation", "performance"):
            with self.subTest(stage=stage):
                metadata = _stage_metadata(run, stage)
                self.assertEqual(metadata["reused_from_experiment"], first_run.experiment_id)
                self.assertEqual(metadata["reused_from"], "attempt-0001")
                self.assertTrue(metadata["reuse_key"].startswith("sha256:"))
        self.assertFalse((run.directory / "checkpoints").exists())
        self.assertFalse((run.directory / "evaluation").exists())
        self.assertEqual(model, first_model)
        self.assertEqual(scientific, first_scientific)
        for stage in ("accuracy_gate", "visualization"):
            self.assertNotIn("reused_from", _stage_metadata(run, stage))
        self.assertIs(acceptance.decision, qec.GateDecision.PASS)
        self.assertIn("paired-relative-non-inferiority", acceptance.rationale)
        self.assertEqual(acceptance.evidence_artifact_ids[0], first_scientific.result_artifact_id)
        self.assertTrue(
            (self.root / "runs" / run.experiment_id / "artifacts" / "attempt-0001.accuracy-gate.json").is_file()
        )
        self.assertIsNotNone(performance)
        self.assertEqual(len(figures), 3)

        again = run_workflow(self.runtime, relative)[0]
        self.assertEqual(again.attempt_id, "attempt-0002")
        training = _stage_metadata(again, "training")
        self.assertEqual(training["reused_from"], "attempt-0001")
        self.assertNotIn("reused_from_experiment", training)
        self.assertNotIn("reuse_key", training)

    def test_cross_experiment_reuse_follows_the_reuse_key(self) -> None:
        first_run, first_model, *_ = run_workflow(self.runtime, tiny_config())
        renamed = run_workflow(self.runtime, tiny_config(**{"experiment.name": "renamed"}))
        for stage in ("training", "scientific_evaluation", "performance"):
            with self.subTest(stage=stage):
                self.assertEqual(
                    _stage_metadata(renamed[0], stage)["reused_from_experiment"],
                    first_run.experiment_id,
                )
        self.assertEqual(renamed[1], first_model)

        longer = run_workflow(self.runtime, tiny_config(**{"training.epochs": 4}))[0]
        training = _stage_metadata(longer, "training")
        self.assertNotIn("reused_from", training)
        self.assertEqual(training["epochs_trained_in_attempt"], 4)

    def test_modified_source_model_is_not_reused_across_experiments(self) -> None:
        _, model, *_ = run_workflow(self.runtime, tiny_config())
        path = self.runtime.artifact_path(_checkpoint(model))
        path.write_bytes(path.read_bytes() + b"tampered")
        run = run_workflow(self.runtime, tiny_config(**{"experiment.name": "renamed"}))[0]
        training = _stage_metadata(run, "training")
        self.assertNotIn("reused_from", training)
        self.assertEqual(training["epochs_trained_in_attempt"], 3)

    def test_preflight_failures_create_nothing(self) -> None:
        qec = self.qec
        cases = {
            "unresolved": (
                tiny_config(**{"model.family": "unresolved"}),
                qec.UnresolvedConfigurationError,
            ),
            "unregistered circuit adapter": (
                tiny_config(**{"qec.circuit_adapter": "qiskit-circuit"}),
                qec.ConfigurationError,
            ),
            "generator version": (
                tiny_config(**{"dataset.generator_version": "0.0.1"}),
                qec.ConfigurationError,
            ),
            "unsupported noise": (
                tiny_config(
                    **{"noise.family": "independent-phase-flip", "noise.time_dependent": True}
                ),
                qec.ConfigurationError,
            ),
            "distributed": (
                tiny_config(**{"execution.distributed": True}),
                qec.ExecutionConfigurationError,
            ),
            "unknown key": (tiny_config(**{"training.warmup": 3}), qec.ConfigurationError),
            "gate baseline": (
                tiny_config(**{"accuracy_gate.baseline_decoder": "other"}),
                qec.ConfigurationError,
            ),
            "negative relative tolerance": (
                tiny_config(
                    **{
                        "accuracy_gate.comparison_rule": "paired-relative-non-inferiority",
                        "accuracy_gate.tolerance": -0.1,
                    }
                ),
                qec.ConfigurationError,
            ),
        }
        for name, (config, error) in cases.items():
            with self.subTest(case=name), self.assertRaises(error):
                self.runtime.create_experiment(config)
        cuda = tiny_config(**{"execution.device": "cuda", "execution.gpu_count": 1})
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(qec.ExecutionConfigurationError, "refusing to fall back to CPU"),
        ):
            self.runtime.create_experiment(cuda)
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "datasets").exists())

    def test_running_attempt_handling(self) -> None:
        qec = self.qec
        experiment = self.runtime.create_experiment(tiny_config())
        abandoned = experiment.start_or_recover()
        resumed = experiment.start_or_recover()
        self.assertIs(abandoned.status, qec.AttemptStatus.INTERRUPTED)
        self.assertEqual(resumed._attempt.recovery_from["terminal_status"], "interrupted")
        with self.assertRaises(qec.AttemptStateError):
            abandoned.resolve_dataset()

        import os

        record = resumed._attempt.record
        record["owner"]["pid"] = os.getppid()
        resumed._attempt._write(record)
        with self.assertRaises(qec.AttemptInProgressError):
            experiment.start_or_recover()

    def test_gate_fail_blocks_performance_and_timeouts_follow_policy(self) -> None:
        qec = self.qec
        config = tiny_config(
            **{
                "accuracy_gate.tolerance": -1.0,
                "model.decoding": {"burn_in": 1, "max_steps": 2},
            }
        )
        run, model, scientific, acceptance, performance, _ = run_workflow(self.runtime, config)
        self.assertIs(acceptance.decision, qec.GateDecision.FAIL)
        self.assertIsNone(performance)
        self.assertEqual(_stage_statuses(run)["performance"], None)
        rbm = scientific.decoder_results[0]
        self.assertGreater(rbm.timeout_count, 0)
        self.assertGreaterEqual(rbm.failure_count, rbm.timeout_count)

        strict = tiny_config(
            **{
                "model.decoding": {"burn_in": 1, "max_steps": 2},
                "scientific_evaluation.invalid_sample_policy": "fail",
            }
        )
        run = self.runtime.create_experiment(strict).start_or_recover()
        dataset = run.resolve_dataset()
        model = run.train(dataset)
        with self.assertRaisesRegex(qec.EvaluationError, "invalid_sample_policy is 'fail'"):
            run.evaluate_accuracy(
                model=model, dataset=dataset, baselines=("pymatching-cpu-decoder",)
            )
        self.assertIs(run.status, qec.AttemptStatus.FAILED)
        with self.assertRaises(qec.AttemptStateError):
            run.evaluate_performance(model=model, dataset=dataset)


def _checkpoint(model):
    from ai_qec.paper.local_runtime import _checkpoint_ref

    return _checkpoint_ref(model)


if __name__ == "__main__":
    unittest.main()
