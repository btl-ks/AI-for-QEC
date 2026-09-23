import subprocess
import sys
import unittest

import ai_qec.notebook_api as qec
from tests.helpers import requires_runtime


class RegistryTests(unittest.TestCase):
    def test_decorator_registration_and_build(self) -> None:
        registry = qec.Registry[dict[str, object]]("dataset.generator")

        @registry.register("stim")
        def build_stim(**kwargs: object) -> dict[str, object]:
            return {"backend": "stim", **kwargs}

        self.assertIs(registry.require("stim"), build_stim)
        self.assertEqual(
            registry.build("stim", distance=3),
            {"backend": "stim", "distance": 3},
        )
        self.assertEqual(registry.keys(), ("stim",))

    def test_duplicate_registration_keeps_original(self) -> None:
        registry = qec.Registry[object]("training.optimizer")

        @registry.register("adam")
        def original(**kwargs: object) -> object:
            return kwargs

        with self.assertRaisesRegex(qec.DuplicateRegistrationError, "duplicate"):

            @registry.register("adam")
            def replacement(**kwargs: object) -> object:
                return None

        self.assertIs(registry.require("adam"), original)

    def test_unknown_key_lists_available_options(self) -> None:
        registry = qec.Registry[object]("qec.code_family")

        @registry.register("surface")
        def surface(**kwargs: object) -> object:
            return kwargs

        with self.assertRaisesRegex(
            qec.UnknownRegistrationError,
            r"qec\.code_family.*toric.*surface",
        ):
            registry.build("toric")

    def test_unresolved_detection_in_mappings_and_sequences(self) -> None:
        config = {
            "dataset": {"generator": "stim", "generator_version": "unresolved"},
            "models": [{"family": "unresolved"}],
            "training": {"optimizer": "adam"},
        }
        self.assertEqual(
            tuple(qec.find_unresolved(config)),
            ("dataset.generator_version", "models[0].family"),
        )

        with self.assertRaises(qec.UnresolvedConfigurationError) as caught:
            qec.validate_no_unresolved(
                config,
                allow={"dataset.generator_version"},
            )
        self.assertEqual(caught.exception.paths, ("models[0].family",))

    def test_unknown_selection_fails_before_factory_call(self) -> None:
        calls: list[str] = []
        registry = qec.Registry[object]("dataset.generator")

        @registry.register("stim")
        def stim(**kwargs: object) -> object:
            calls.append("stim")
            return kwargs

        config = {"dataset": {"generator": "cuda-q"}}
        with self.assertRaisesRegex(qec.ConfigurationError, "cuda-q"):
            qec.validate_config(
                config,
                required_selections={"dataset.generator"},
                registries={"dataset.generator": registry},
            )
        self.assertEqual(calls, [])

    def test_build_from_config_uses_declared_path(self) -> None:
        registry = qec.Registry[object]("dataset.generator")

        @registry.register("stim")
        def stim(**kwargs: object) -> object:
            return kwargs

        result = qec.build_from_config(
            {"dataset": {"generator": "stim"}},
            "dataset.generator",
            registries={"dataset.generator": registry},
            shots=128,
        )
        self.assertEqual(result, {"shots": 128})

    def test_global_registry_paths_are_unique_and_have_no_fake_factories(self) -> None:
        self.assertEqual(len(qec.REGISTRIES_BY_PATH), 14)
        self.assertEqual(len(set(qec.REGISTRIES_BY_PATH)), 14)
        self.assertEqual(qec.CODES.name, "qec.code_family")
        self.assertEqual(qec.NOISE.name, "noise.family")
        self.assertEqual(qec.GENERATORS.name, "dataset.generator")
        self.assertEqual(qec.OPTIMIZERS.name, "training.optimizer")
        self.assertEqual(qec.SCHEDULERS.name, "training.scheduler")
        self.assertEqual(qec.LOSSES.name, "training.loss")
        self.assertEqual(qec.TRAINING_STEP_EXECUTORS.name, "execution.step_executor")
        self.assertEqual(
            qec.CPU_TO_GPU_PIPELINES.name,
            "data_pipeline.cpu_to_gpu.technology",
        )
        self.assertEqual(
            qec.GPU_TO_GPU_PIPELINES.name,
            "data_pipeline.gpu_to_gpu.technology",
        )

    def test_importing_the_facade_registers_nothing(self) -> None:
        script = (
            "import ai_qec.notebook_api as qec; "
            "print(sum(len(registry) for registry in qec.REGISTRIES_BY_PATH.values()))"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "0")

    @requires_runtime
    def test_loaded_implementations_are_exactly_the_executable_ones(self) -> None:
        from ai_qec.registry.bootstrap import load_builtin_implementations

        registered = load_builtin_implementations()
        self.assertEqual(
            {path: keys for path, keys in registered.items() if keys},
            {
                "qec.code_family": ("toric",),
                "noise.family": ("independent-phase-flip",),
                "noise.adapter": ("stim-noise",),
                "dataset.generator": ("stim-syndrome-cpu",),
                "model.family": ("joint-error-syndrome-rbm",),
                "training.optimizer": ("sgd",),
                "training.scheduler": ("constant",),
                "training.loss": ("contrastive-divergence",),
                "execution.trainer_framework": ("pytorch",),
                "execution.step_executor": ("pytorch-cuda-graph", "pytorch-eager"),
                "scientific_evaluation.baseline_decoders": ("pymatching-cpu-decoder",),
                "data_pipeline.cpu_to_gpu.technology": ("pytorch-dataloader-h2d",),
            },
        )
        self.assertNotIn("qiskit-circuit", qec.CIRCUITS)
        self.assertEqual(len(qec.GPU_TO_GPU_PIPELINES), 0)
        self.assertEqual(load_builtin_implementations(), registered)


if __name__ == "__main__":
    unittest.main()
