import unittest

from tests.helpers import requires_runtime


@requires_runtime
class StimGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        import stim

        import ai_qec.notebook_api as qec
        from ai_qec.data.generators.stim_code_capacity import build_stim_code_capacity_generator
        from ai_qec.qec.backends.stim_noise import StimNoiseCompiler
        from ai_qec.qec.codes.toric import ToricCode

        self.qec = qec
        self.code = ToricCode(4)
        self.noise = qec.NoiseSpec("independent-phase-flip", {"physical_error_rate": 0.1})
        self.compilation = StimNoiseCompiler().compile(self.noise)
        self.spec = qec.DatasetSpec(
            qec=qec.QECSpec("toric", 4, 1, "X", "code-capacity"),
            noise=self.noise,
            generator="stim-syndrome-cpu",
            generator_version=stim.__version__,
            backend_semantics="exact",
            train_samples=3000,
            validation_samples=100,
            test_samples=100,
            seed=3,
            split_policy="independent-streams",
            schema_version="qec-batch-v1",
        )
        self.build = lambda **kwargs: build_stim_code_capacity_generator(
            code=self.code, noise=self.compilation, dataset_artifact_id="ds-test", **kwargs
        )

    def test_samples_match_code_definition_and_noise_rate(self) -> None:
        import numpy as np

        from ai_qec.data.datasets.local import code_consistency_validator

        validate = code_consistency_validator(self.code)
        batches = list(self.build().generate(self.spec))
        self.assertEqual(
            [batch.context["split"] for batch in batches], ["train", "validation", "test"]
        )
        for batch in batches:
            validate(batch)
            self.assertEqual(batch.dataset_artifact_id, "ds-test")
        weight = batches[0].physical_errors.mean()
        self.assertAlmostEqual(float(weight), 0.1, delta=0.01)
        self.assertEqual(batches[0].sample_ids[0], "train-0000000")
        self.assertTrue(
            np.array_equal(
                batches[0].detector_events, self.code.syndrome(batches[0].physical_errors)
            )
        )

    def test_same_seed_reproduces_and_splits_are_independent(self) -> None:
        import numpy as np

        first = list(self.build().generate(self.spec))
        second = list(self.build().generate(self.spec))
        for left, right in zip(first, second):
            self.assertTrue(np.array_equal(left.physical_errors, right.physical_errors))
        self.assertFalse(np.array_equal(first[1].physical_errors, first[2].physical_errors))

    def test_unsupported_semantics_are_rejected_before_sampling(self) -> None:
        import dataclasses

        from ai_qec.data.generators.stim_code_capacity import GeneratorCompatibilityError

        cases = {
            "generator_version": dataclasses.replace(self.spec, generator_version="0.0.1"),
            "logical_basis": dataclasses.replace(
                self.spec, qec=dataclasses.replace(self.spec.qec, logical_basis="Z")
            ),
            "rounds": dataclasses.replace(
                self.spec, qec=dataclasses.replace(self.spec.qec, rounds=3)
            ),
            "backend_semantics": dataclasses.replace(self.spec, backend_semantics="approximate"),
        }
        for field, spec in cases.items():
            with self.subTest(field=field):
                compatibility = self.build().compatibility(spec)
                self.assertIs(compatibility.support, self.qec.NoiseApproximation.UNSUPPORTED)
                self.assertIn(field.split("_")[0], compatibility.reason)
                with self.assertRaises(GeneratorCompatibilityError):
                    next(iter(self.build().generate(spec)))

    def test_noise_compiler_is_exact_or_explicitly_unsupported(self) -> None:
        from ai_qec.qec.backends.stim_noise import StimNoiseCompiler

        self.assertIs(self.compilation.support, self.qec.NoiseApproximation.EXACT)
        self.assertEqual(self.compilation.model.instruction, "Z_ERROR")
        rejected = StimNoiseCompiler().compile(
            self.qec.NoiseSpec("depolarizing", {"physical_error_rate": 0.1})
        )
        self.assertIs(rejected.support, self.qec.NoiseApproximation.UNSUPPORTED)
        self.assertIsNone(rejected.model)
        with self.assertRaises(self.qec.ConfigurationError):
            StimNoiseCompiler().compile(
                self.qec.NoiseSpec("independent-phase-flip", {"physical_error_rate": 0.7})
            )


if __name__ == "__main__":
    unittest.main()
