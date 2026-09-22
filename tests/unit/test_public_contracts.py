from dataclasses import FrozenInstanceError
import unittest

import ai_qec.notebook_api as qec


class PublicContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qec_spec = qec.QECSpec(
            code_family="toric",
            distance=3,
            rounds=1,
            logical_basis="Z",
            circuit_family="code-capacity",
        )
        self.noise_spec = qec.NoiseSpec(
            family="independent-phase-flip",
            parameters={"physical_error_rate": 0.05},
        )
        self.dataset_spec = qec.DatasetSpec(
            qec=self.qec_spec,
            noise=self.noise_spec,
            generator="contract-only",
            generator_version="0",
            backend_semantics="exact",
            train_samples=32,
            validation_samples=16,
            test_samples=16,
            seed=7,
            split_policy="fixed",
        )

    def test_all_public_names_are_importable(self) -> None:
        for name in qec.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(qec, name))

    def test_specs_are_immutable_value_objects(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.qec_spec.distance = 5  # type: ignore[misc]

    def test_dataset_three_layer_contracts_are_distinct(self) -> None:
        key = qec.DatasetKey(value="sha256:example", algorithm="sha256")
        split = qec.DatasetSplit(
            name="test",
            sample_count=16,
            shard_uris=("file:///dataset/test/shard-00000.bin",),
            checksum="sha256:split",
        )
        artifact = qec.DatasetArtifact(
            artifact_id="dataset-example",
            key=key,
            manifest_uri="file:///dataset/manifest.json",
            checksum="sha256:artifact",
            splits=(split,),
            generator_provenance_uri="file:///dataset/generation.json",
        )
        instance = qec.DatasetInstance(
            instance_id="instance-1",
            attempt_id="attempt-1",
            dataset_artifact_id=artifact.artifact_id,
            roles=(qec.DatasetRole.SCIENTIFIC_EVALUATION,),
            resolved_from_cache=True,
        )

        self.assertIsInstance(self.dataset_spec, qec.DatasetSpec)
        self.assertEqual(instance.dataset_artifact_id, artifact.artifact_id)
        self.assertNotEqual(type(self.dataset_spec), type(artifact))
        self.assertNotEqual(type(artifact), type(instance))

    def test_model_and_recovery_checkpoints_are_distinct(self) -> None:
        model = qec.ModelCheckpoint(
            checkpoint_id="model-1",
            artifact_id="artifact-model-1",
            model_identity="model-digest",
            dataset_artifact_id="dataset-example",
            uri="file:///checkpoints/model.pt",
            checksum="sha256:model",
        )
        recovery = qec.TrainingRecoveryCheckpoint(
            checkpoint_id="recovery-1",
            artifact_id="artifact-recovery-1",
            source_attempt_id="attempt-1",
            epoch=3,
            global_step=42,
            uri="file:///checkpoints/recovery.pt",
            checksum="sha256:recovery",
            includes_optimizer=True,
            includes_scheduler=True,
            includes_amp_scaler=False,
            includes_rng_state=True,
            includes_data_cursor=True,
        )

        self.assertNotEqual(type(model), type(recovery))
        self.assertEqual(recovery.source_attempt_id, "attempt-1")

    def test_scientific_result_requires_ler_evidence(self) -> None:
        estimate = qec.MetricEstimate(
            name="logical_error_rate",
            value=0.1,
            numerator=10,
            denominator=100,
            confidence_level=0.95,
            interval_low=0.055,
            interval_high=0.174,
            method="wilson",
        )
        decoder_result = qec.DecoderEvaluation(
            decoder_id="mwpm",
            dataset_artifact_id="dataset-example",
            sample_ids_digest="sha256:samples",
            logical_error_rate=estimate,
            failure_count=10,
            timeout_count=0,
            not_converged_count=0,
            metrics_artifact_id="metrics-1",
        )

        self.assertEqual(decoder_result.logical_error_rate.denominator, 100)
        self.assertEqual(qec.GateDecision.PASS.value, "pass")

    def test_contract_extensions_keep_backward_compatible_defaults(self) -> None:
        training = qec.TrainingSpec("sgd", 0.1, 3, 32, "constant", "contrastive-divergence")
        self.assertEqual(dict(training.optimizer_parameters), {})
        self.assertEqual(dict(training.loss_parameters), {})
        estimate = qec.MetricEstimate("logical_error_rate", 0.1, 1, 10, 0.95, 0.0, 0.4, "wilson")
        evaluation = qec.DecoderEvaluation("mwpm", "ds", "sha256:x", estimate, 1, 0, 0, "metrics-1")
        self.assertEqual(dict(evaluation.logical_class_counts), {})

    def test_facade_publishes_only_the_real_runtime_entrypoint(self) -> None:
        self.assertTrue(callable(qec.LocalNotebookPlatform))
        self.assertIsNotNone(qec.NotebookPlatform)

    def test_scaffold_does_not_publish_fake_execution_entrypoints(self) -> None:
        self.assertFalse(hasattr(qec, "create_experiment"))
        self.assertFalse(hasattr(qec, "train"))
        self.assertFalse(hasattr(qec, "generate_dataset"))


if __name__ == "__main__":
    unittest.main()
