import unittest

import ai_qec.notebook_api as qec
from ai_qec.experiment.config import parse_experiment_config
from ai_qec.experiment.reuse import stage_reuse_key
from tests.helpers import tiny_config


class _Loss:
    def __init__(self) -> None:
        self.backward_calls = 0

    def backward(self) -> None:
        self.backward_calls += 1

    def detach(self):
        return self


class _Optimizer:
    def __init__(self) -> None:
        self.zero_calls = []
        self.step_calls = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.zero_calls.append(set_to_none)

    def step(self) -> None:
        self.step_calls += 1


class TrainingStepExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import importlib

        importlib.import_module("ai_qec.training.executors.eager")
        importlib.import_module("ai_qec.training.executors.cuda_graph")

    def test_registry_contains_only_real_step_factories(self) -> None:
        self.assertEqual(
            qec.TRAINING_STEP_EXECUTORS.keys(),
            ("pytorch-cuda-graph", "pytorch-eager"),
        )
        for key in qec.TRAINING_STEP_EXECUTORS:
            plan = qec.TRAINING_STEP_EXECUTORS.build(
                key,
                device="cpu" if key == "pytorch-eager" else "cuda:0",
                options={} if key == "pytorch-eager" else {"max_graphs": 2},
            )
            self.assertIsInstance(plan, qec.TrainingStepPlan)

    def test_eager_executor_performs_exactly_one_update(self) -> None:
        optimizer = _Optimizer()
        loss = _Loss()
        calls = []

        def visible(tensors):
            calls.append(("visible", tensors))
            return "visible"

        def objective(model, value, generator):
            calls.append(("objective", model, value, generator))
            return loss

        executor = qec.TRAINING_STEP_EXECUTORS.build(
            "pytorch-eager",
            device="cpu",
            options={},
            context=qec.TrainingStepContext(
                model="model",
                visible=visible,
                objective=objective,
                optimizer=optimizer,
                generator="rng",
                device="cpu",
            ),
        )
        result = executor.step({"x": "batch"})
        executor.finalize()

        self.assertEqual(result.execution_mode, "eager")
        self.assertEqual(loss.backward_calls, 1)
        self.assertEqual(optimizer.zero_calls, [True])
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(calls[0], ("visible", {"x": "batch"}))
        self.assertFalse(executor.evidence().fallback_observed)

    def test_options_are_strict_and_graph_requires_cuda(self) -> None:
        cases = (
            ("pytorch-eager", "cpu", {"max_graphs": 1}),
            ("pytorch-cuda-graph", "cpu", {"max_graphs": 1}),
            ("pytorch-cuda-graph", "cuda:0", {}),
            ("pytorch-cuda-graph", "cuda:0", {"max_graphs": 0}),
            ("pytorch-cuda-graph", "cuda:0", {"max_graphs": True}),
            ("pytorch-cuda-graph", "cuda:0", {"max_graphs": 1, "extra": 2}),
        )
        for executor, device, options in cases:
            with (
                self.subTest(executor=executor, options=options),
                self.assertRaises(qec.ConfigurationError),
            ):
                qec.TRAINING_STEP_EXECUTORS.build(executor, device=device, options=options)

    def test_unknown_and_unresolved_selection_fail_validation(self) -> None:
        for selected in ("unknown", "unresolved"):
            config = tiny_config(**{"execution.step_executor": selected})
            with self.subTest(selected=selected), self.assertRaises(qec.ConfigurationError):
                qec.validate_config(config, required_selections=("execution.step_executor",))

    def test_executor_changes_experiment_and_training_not_dataset(self) -> None:
        eager_config = tiny_config()
        graph_config = tiny_config(
            **{
                "execution.device": "cuda",
                "execution.gpu_count": 1,
                "execution.step_executor": "pytorch-cuda-graph",
                "execution.step_executor_options": {"max_graphs": 2},
            }
        )
        eager = parse_experiment_config(eager_config)
        graph = parse_experiment_config(graph_config)
        self.assertEqual(eager.spec.dataset, graph.spec.dataset)
        self.assertNotEqual(eager.digest, graph.digest)
        dataset = qec.ArtifactRef(
            "ds",
            qec.ArtifactKind.DATASET,
            "datasets/ds/manifest.json",
            "sha256:d",
            "application/json",
        )
        self.assertNotEqual(
            stage_reuse_key("training", eager_config, (dataset,)),
            stage_reuse_key("training", graph_config, (dataset,)),
        )

    def test_execution_section_rejects_unknown_key(self) -> None:
        with self.assertRaisesRegex(qec.ConfigurationError, "unknown keys"):
            parse_experiment_config(tiny_config(**{"execution.graph_pool": "global"}))


if __name__ == "__main__":
    unittest.main()
