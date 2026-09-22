import unittest
from itertools import pairwise

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_CUDA = False


@unittest.skipUnless(HAS_CUDA, "requires a CUDA-capable PyTorch runtime")
class CUDAGraphExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        import ai_qec.notebook_api as qec
        import ai_qec.training.executors.cuda_graph as cuda_graph

        self.qec = qec
        self.cuda_graph = cuda_graph

    def _executor(self, *, max_graphs=2, random=False, objective=None):
        model = torch.nn.Linear(3, 1, bias=False, device="cuda")
        torch.nn.init.constant_(model.weight, 0.25)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        generator = torch.Generator(device="cuda").manual_seed(123)

        def default_objective(network, visible, rng):
            prediction = network(visible["x"])
            if random:
                prediction = prediction + torch.rand(prediction.shape, device="cuda", generator=rng)
            return torch.square(prediction - visible["target"]).mean()

        context = self.qec.TrainingStepContext(
            model=model,
            visible=lambda tensors: tensors,
            objective=objective or default_objective,
            optimizer=optimizer,
            generator=generator,
            device="cuda:0",
        )
        executor = self.cuda_graph.build_pytorch_cuda_graph_training_step_executor(
            device="cuda:0", options={"max_graphs": max_graphs}, context=context
        )
        return executor, model, generator

    @staticmethod
    def _batch(rows, value):
        return {
            "x": torch.full((rows, 3), float(value), device="cuda"),
            "target": torch.full((rows, 1), 1.0 + float(value) / 10.0, device="cuda"),
        }

    def test_static_input_uses_each_new_batch(self) -> None:
        executor, _, _ = self._executor()
        losses = []
        for value in (1.0, 2.0, -3.0):
            result = executor.step(self._batch(4, value))
            torch.cuda.synchronize()
            losses.append(float(result.loss.detach().cpu()))
        executor.finalize()

        self.assertNotEqual(losses[1], losses[2])
        state = next(iter(executor._states.values()))
        self.assertEqual(tuple(state.static_inputs), ("target", "x"))
        self.assertEqual(sum(t.numel() for t in state.static_inputs.values()), 16)
        self.assertEqual(executor.evidence().replay_steps, 1)

    def test_warmup_capture_and_replay_count_exact_updates(self) -> None:
        executor, model, _ = self._executor()
        weights = [model.weight.detach().clone()]
        modes = []
        for value in (1.0, 2.0, 3.0, 4.0, 5.0):
            result = executor.step(self._batch(4, value))
            torch.cuda.synchronize()
            modes.append(result.execution_mode)
            weights.append(model.weight.detach().clone())
        executor.finalize()

        evidence = executor.evidence()
        self.assertEqual(modes[:2], ["cuda-graph-warmup", "cuda-graph-capture"])
        self.assertEqual(modes[2:], ["cuda-graph-replay"] * 3)
        self.assertEqual(
            evidence.warmup_steps + evidence.capture_count + evidence.replay_steps,
            5,
        )
        self.assertEqual(
            (evidence.warmup_steps, evidence.capture_count, evidence.replay_steps),
            (1, 1, 3),
        )
        self.assertTrue(all(not torch.equal(a, b) for a, b in pairwise(weights)))
        self.assertGreaterEqual(evidence.capture_seconds, 0.0)
        self.assertFalse(evidence.fallback_observed)

    def test_alternating_signatures_and_limit_before_update(self) -> None:
        executor, _, _ = self._executor(max_graphs=2)
        for rows, value in ((4, 1), (2, 1), (4, 2), (2, 2), (4, 3), (2, 3)):
            executor.step(self._batch(rows, value))
        torch.cuda.synchronize()
        executor.finalize()
        evidence = executor.evidence()
        self.assertEqual(len(evidence.signatures), 2)
        self.assertEqual(
            (evidence.warmup_steps, evidence.capture_count, evidence.replay_steps),
            (2, 2, 2),
        )
        self.assertEqual(
            {item.signature.tensors[0].shape[0] for item in evidence.signatures},
            {2, 4},
        )

        limited, model, _ = self._executor(max_graphs=1)
        limited.step(self._batch(4, 1))
        torch.cuda.synchronize()
        before = model.weight.detach().clone()
        with self.assertRaisesRegex(self.cuda_graph.CUDAGraphExecutionError, "max_graphs=1"):
            limited.step(self._batch(2, 2))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(before, model.weight))

    def test_capture_failure_has_no_fallback(self) -> None:
        def unsafe_objective(network, visible, generator):
            scale = float(visible["x"].sum().item())
            return torch.square(network(visible["x"]) - scale).mean()

        executor, _, _ = self._executor(objective=unsafe_objective)
        executor.step(self._batch(4, 1))
        with self.assertRaises(self.cuda_graph.CUDAGraphExecutionError) as caught:
            executor.step(self._batch(4, 2))
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertFalse(caught.exception.evidence.fallback_observed)
        self.assertEqual(caught.exception.evidence.capture_count, 0)

    def test_replay_advances_registered_generator(self) -> None:
        executor, model, generator = self._executor(random=True)
        for value in (1.0, 2.0):
            executor.step(self._batch(4, value))
        torch.cuda.synchronize()
        generator_states = [generator.get_state().clone()]
        weights = [model.weight.detach().clone()]
        for value in (3.0, 4.0):
            executor.step(self._batch(4, value))
            torch.cuda.synchronize()
            generator_states.append(generator.get_state().clone())
            weights.append(model.weight.detach().clone())
        executor.finalize()

        self.assertTrue(
            all(not torch.equal(first, second) for first, second in pairwise(generator_states))
        )
        self.assertTrue(all(not torch.equal(first, second) for first, second in pairwise(weights)))


if __name__ == "__main__":
    unittest.main()
