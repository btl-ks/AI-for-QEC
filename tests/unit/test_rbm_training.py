import itertools
import unittest

from tests.helpers import requires_runtime


@requires_runtime
class JointRBMTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        from ai_qec.models.generative.rbm import JointErrorSyndromeRBMFamily
        from ai_qec.models.spec import ModelSpec

        self.torch = torch
        self.family = JointErrorSyndromeRBMFamily()
        self.spec = ModelSpec("joint-error-syndrome-rbm", "1", {"hidden_units": 3, "init_std": 0.5})
        self.model = self.family.create(
            self.spec, widths={"physical_errors": 3, "detector_events": 2}, seed=7
        )
        with torch.no_grad():
            self.model.visible_bias.copy_(torch.tensor([0.3, -0.2, 0.1, 0.4, -0.5]))
            self.model.hidden_bias.copy_(torch.tensor([-0.1, 0.2, 0.05]))

    def _energy(self, visible, hidden):
        model = self.model
        return (
            -(hidden @ model.weight @ visible)
            - model.visible_bias @ visible
            - model.hidden_bias @ hidden
        )

    def test_free_energy_matches_enumeration_of_hidden_units(self) -> None:
        torch = self.torch
        for bits in itertools.product((0.0, 1.0), repeat=5):
            visible = torch.tensor(bits)
            hiddens = [torch.tensor(h) for h in itertools.product((0.0, 1.0), repeat=3)]
            exact = -torch.logsumexp(
                torch.stack([-self._energy(visible, hidden) for hidden in hiddens]), dim=0
            )
            self.assertAlmostEqual(
                float(self.model.free_energy(visible[None, :])[0]), float(exact), places=5
            )

    def test_conditionals_match_the_joint_distribution(self) -> None:
        torch = self.torch
        visible = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
        hiddens = [torch.tensor(h) for h in itertools.product((0.0, 1.0), repeat=3)]
        weights = torch.stack([torch.exp(-self._energy(visible, hidden)) for hidden in hiddens])
        marginal = (weights[:, None] * torch.stack(hiddens)).sum(0) / weights.sum()
        self.assertTrue(
            torch.allclose(self.model.hidden_probability(visible[None, :])[0], marginal, atol=1e-6)
        )
        self.assertEqual(self.model.error_weight.shape, (3, 3))
        self.assertEqual(self.model.syndrome_weight.shape, (3, 2))

    def test_payload_round_trip_and_parameter_validation(self) -> None:
        torch = self.torch
        from ai_qec.registry.validation import ConfigurationError
        from ai_qec.models.spec import ModelSpec

        payload = self.family.payload(
            self.model,
            self.spec,
            widths={"physical_errors": 3, "detector_events": 2},
            dataset_artifact_id="ds-x",
        )
        loaded = self.family.load(payload, device="cpu")
        for name, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[name], tensor))
        for parameters in (
            {"hidden_units": 0, "init_std": 0.1},
            {"hidden_units": 4},
            {"hidden_units": 4, "init_std": 0.1, "x": 1},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ConfigurationError):
                self.family.parameters(ModelSpec("joint-error-syndrome-rbm", "1", parameters))
        with self.assertRaises(ConfigurationError):
            self.family.parameters(
                ModelSpec("joint-error-syndrome-rbm", "2", {"hidden_units": 4, "init_std": 0.1})
            )
        with self.assertRaises(ConfigurationError):
            self.family.decoding_parameters({"burn_in": 10, "max_steps": 10})

    def test_contrastive_divergence_gradient_lowers_free_energy_of_data(self) -> None:
        torch = self.torch
        from ai_qec.registry.bootstrap import load_builtin_implementations
        from ai_qec.registry.catalog import LOSSES, OPTIMIZERS

        load_builtin_implementations()
        objective = LOSSES.build("contrastive-divergence", parameters={"cd_steps": 1})
        optimizer = OPTIMIZERS.build(
            "sgd", parameters=self.model.parameters(), learning_rate=0.1, options={}
        )
        data = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]] * 32)
        generator = torch.Generator().manual_seed(0)
        before = float(self.model.free_energy(data).mean())
        for _ in range(50):
            loss = objective(self.model, data, generator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(float(self.model.free_energy(data).mean()), before)


if __name__ == "__main__":
    unittest.main()
