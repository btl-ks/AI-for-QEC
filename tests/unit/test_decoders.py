import unittest

from tests.helpers import requires_runtime


def _batch(syndromes, truth):
    import ai_qec.notebook_api as qec

    return qec.QECBatch(
        detector_events=syndromes,
        observable_truth=truth,
        sample_ids=tuple(f"test-{index:07d}" for index in range(len(syndromes))),
        dataset_artifact_id="ds-test",
        layout=qec.BatchLayout(
            qec.BatchRepresentation.NUMPY_ARRAY,
            qec.MemoryResidency.HOST,
            "cpu",
            "uint8",
            syndromes.shape,
        ),
    )


@requires_runtime
class DecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        import ai_qec.notebook_api as qec
        from ai_qec.data.loaders.pytorch import build_torch_h2d_pipeline
        from ai_qec.models.decoders.classical.pymatching_adapter import PyMatchingDecoder
        from ai_qec.models.generative.rbm import JointErrorSyndromeRBMFamily
        from ai_qec.qec.codes.toric import ToricCode

        self.np, self.qec = np, qec
        self.code = ToricCode(3)
        rng = np.random.default_rng(0)
        self.errors = (rng.random((40, self.code.num_data_qubits)) < 0.08).astype(np.uint8)
        self.batch = _batch(self.code.syndrome(self.errors), self.code.logical_flips(self.errors))
        self.pymatching = PyMatchingDecoder(code=self.code)
        family = JointErrorSyndromeRBMFamily()
        model = family.create(
            qec.ModelSpec("joint-error-syndrome-rbm", "1", {"hidden_units": 8, "init_std": 1e-6}),
            widths={
                "physical_errors": self.code.num_data_qubits,
                "detector_events": self.code.num_checks,
            },
            seed=0,
        )
        pipeline = build_torch_h2d_pipeline(spec=qec.CPUToGPUPipelineSpec(), device="cpu")
        self.rbm = lambda burn_in, max_steps: family.build_decoder(
            model,
            code=self.code,
            decoding={"burn_in": burn_in, "max_steps": max_steps},
            pipeline=pipeline,
            device="cpu",
            seed=1,
        )

    def _request(self, batch, decoder_id):
        return self.qec.DecodeRequest(request_id="r1", decoder_id=decoder_id, batch=batch)

    def test_pymatching_corrects_every_single_qubit_error(self) -> None:
        np = self.np
        errors = np.eye(self.code.num_data_qubits, dtype=np.uint8)
        batch = _batch(self.code.syndrome(errors), self.code.logical_flips(errors))
        result = self.pymatching.decode(self._request(batch, self.pymatching.decoder_id))
        self.assertIs(result.status, self.qec.DecodeStatus.SUCCEEDED)
        self.assertTrue(np.array_equal(result.predictions, batch.observable_truth))
        self.assertEqual(result.runtime.technology_id, "pymatching-cpu-decoder")

    def test_accepted_rbm_recoveries_reproduce_the_syndrome(self) -> None:
        np = self.np
        # Near-zero weights make p(e | h) uniform: each Gibbs step proposes a random chain,
        # so every chain eventually satisfies its syndrome.
        decoder = self.rbm(burn_in=1, max_steps=5000)
        result = decoder.decode(self._request(self.batch, decoder.decoder_id))
        self.assertIs(result.status, self.qec.DecodeStatus.SUCCEEDED)
        self.assertEqual(result.failed_sample_ids, ())
        self.assertTrue(np.isin(result.predictions, (0, 1)).all())
        self.assertEqual(result.provenance["accepted"], 40)
        self.assertGreater(result.provenance["acceptance_steps"]["mean"], 1)

    def test_rbm_budget_exhaustion_is_reported_not_predicted(self) -> None:
        np = self.np
        decoder = self.rbm(burn_in=1, max_steps=2)
        result = decoder.decode(self._request(self.batch, decoder.decoder_id))
        self.assertIs(result.status, self.qec.DecodeStatus.TIMED_OUT)
        failed = [int(sample_id.split("-")[1]) for sample_id in result.failed_sample_ids]
        self.assertGreater(len(failed), 30)
        self.assertTrue(np.all(result.predictions[failed] == -1))
        accepted = np.setdiff1d(np.arange(40), failed)
        self.assertTrue(np.isin(result.predictions[accepted], (0, 1)).all())

    def test_width_mismatch_is_unsupported(self) -> None:
        np = self.np
        wrong = _batch(np.zeros((3, 5), dtype=np.uint8), np.zeros((3, 2), dtype=np.uint8))
        for decoder in (self.pymatching, self.rbm(burn_in=1, max_steps=5)):
            with self.subTest(decoder=decoder.decoder_id):
                result = decoder.decode(self._request(wrong, decoder.decoder_id))
                self.assertIs(result.status, self.qec.DecodeStatus.UNSUPPORTED)
                self.assertIsNone(result.predictions)


@requires_runtime
class ExecutionPlannerTests(unittest.TestCase):
    def test_unsupported_options_fail_with_every_reason(self) -> None:
        from unittest import mock

        import ai_qec.notebook_api as qec
        from ai_qec.training.execution_planner import LocalExecutionPlanner

        spec = qec.ExecutionSpec(
            device="cuda",
            cpu_count=1,
            gpu_count=2,
            distributed=True,
            num_workers=0,
            mixed_precision=True,
            compile_model=True,
        )
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            self.assertRaises(qec.ExecutionConfigurationError) as caught,
        ):
            LocalExecutionPlanner().resolve(spec)
        message = str(caught.exception)
        for fragment in (
            "distributed",
            "mixed_precision",
            "compile_model",
            "CUDA is not available",
            "gpu_count",
        ):
            self.assertIn(fragment, message)

    def test_cpu_plan_records_framework_and_environment(self) -> None:
        import ai_qec.notebook_api as qec
        from ai_qec.training.execution_planner import LocalExecutionPlanner

        plan = LocalExecutionPlanner().resolve(
            qec.ExecutionSpec("cpu", 1, 0, False, 0, False, False)
        )
        self.assertEqual(
            (plan.device, plan.world_size, plan.trainer_framework), ("cpu", 1, "pytorch")
        )
        self.assertTrue(plan.environment_digest.startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
