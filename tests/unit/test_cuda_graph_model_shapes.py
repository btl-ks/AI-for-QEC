import unittest

from tests.helpers import requires_runtime

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_CUDA = False


def _fixtures(device):
    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Conv2d(1, 2, 3), torch.nn.Flatten(), torch.nn.Linear(8, 2)
            )

        def forward(self, values):
            return self.network(values["x"])

    class Recurrent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.GRU(5, 7, batch_first=True)
            self.output = torch.nn.Linear(7, 2)

        def forward(self, values):
            sequence, _ = self.rnn(values["x"])
            return self.output(sequence[:, -1])

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = torch.nn.MultiheadAttention(8, 2, batch_first=True)
            self.output = torch.nn.Linear(8, 2)

        def forward(self, values):
            attended, _ = self.attention(values["x"], values["x"], values["x"], need_weights=False)
            return self.output(attended.mean(dim=1))

    class GraphAggregation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Linear(4, 2)

        def forward(self, values):
            aggregated = torch.bmm(values["adjacency"], values["x"])
            return self.output(aggregated.mean(dim=1))

    return (
        (Conv().to(device), {"x": torch.randn(4, 1, 4, 4, device=device)}),
        (Recurrent().to(device), {"x": torch.randn(4, 3, 5, device=device)}),
        (Attention().to(device), {"x": torch.randn(4, 3, 8, device=device)}),
        (
            GraphAggregation().to(device),
            {
                "x": torch.randn(4, 5, 4, device=device),
                "adjacency": torch.eye(5, device=device).expand(4, -1, -1).clone(),
            },
        ),
    )


def _context(qec, model, optimizer, device):
    generator = torch.Generator(device=device).manual_seed(7)

    def objective(network, values, rng):
        return torch.square(network(values)).mean()

    return qec.TrainingStepContext(
        model=model,
        visible=lambda tensors: tensors,
        objective=objective,
        optimizer=optimizer,
        generator=generator,
        device=str(device),
    )


@requires_runtime
class EagerModelShapeTests(unittest.TestCase):
    def test_eager_uses_one_contract_for_conv_rnn_attention_and_graph(self) -> None:
        import ai_qec.notebook_api as qec
        from ai_qec.training.executors.eager import build_pytorch_eager_training_step_executor

        for model, batch in _fixtures(torch.device("cpu")):
            with self.subTest(model=type(model).__name__):
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                executor = build_pytorch_eager_training_step_executor(
                    device="cpu",
                    options={},
                    context=_context(qec, model, optimizer, torch.device("cpu")),
                )
                executor.step(batch)
                executor.finalize()
                self.assertEqual(executor.evidence().observed_executor, "pytorch-eager")


@unittest.skipUnless(HAS_CUDA, "requires a CUDA-capable PyTorch runtime")
class CUDAGraphModelShapeTests(unittest.TestCase):
    def test_cuda_graph_uses_one_contract_for_conv_rnn_attention_and_graph(self) -> None:
        import ai_qec.notebook_api as qec
        from ai_qec.training.executors.cuda_graph import (
            build_pytorch_cuda_graph_training_step_executor,
        )

        device = torch.device("cuda:0")
        for model, batch in _fixtures(device):
            with self.subTest(model=type(model).__name__):
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                executor = build_pytorch_cuda_graph_training_step_executor(
                    device="cuda:0",
                    options={"max_graphs": 1},
                    context=_context(qec, model, optimizer, device),
                )
                for offset in (0.0, 0.01, 0.02):
                    executor.step(
                        {
                            name: value + offset if value.is_floating_point() else value
                            for name, value in batch.items()
                        }
                    )
                torch.cuda.synchronize()
                executor.finalize()
                evidence = executor.evidence()
                self.assertEqual((evidence.capture_count, evidence.replay_steps), (1, 1))
                self.assertFalse(evidence.fallback_observed)

        self.assertFalse(
            any(
                name in qec.MODELS
                for name in ("Conv", "Recurrent", "Attention", "GraphAggregation")
            )
        )


if __name__ == "__main__":
    unittest.main()
