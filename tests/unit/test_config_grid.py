import copy
import unittest

import ai_qec.notebook_api as qec


BASE = {
    "experiment": {"name": "base", "master_seed": 1},
    "qec": {"distance": 4},
    "noise": {"parameters": {"physical_error_rate": 0.08}},
    "model": {"parameters": {"hidden_units": 64}},
    "training": {"epochs": 30},
    "dataset": {"train_samples": 100},
}
AXES = {"L": ("qec.distance", (4, 6)), "p": ("noise.parameters.physical_error_rate", (0.05, 0.1))}
COUPLED = {
    "L": {4: {"model.parameters.hidden_units": 64}, 6: {"model.parameters.hidden_units": 128}}
}


class ConfigGridTests(unittest.TestCase):
    def test_cartesian_product_with_coupled_and_shared_overrides(self) -> None:
        original = copy.deepcopy(BASE)
        points = qec.config_grid(
            BASE,
            name="run-l{L}-p{p:.2f}",
            axes=AXES,
            coupled=COUPLED,
            overrides={"dataset.train_samples": 7},
        )
        self.assertEqual(BASE, original)
        self.assertEqual(
            [point.values for point in points],
            [
                {"L": 4, "p": 0.05},
                {"L": 4, "p": 0.1},
                {"L": 6, "p": 0.05},
                {"L": 6, "p": 0.1},
            ],
        )
        last = points[-1].config
        self.assertEqual(last["experiment"], {"name": "run-l6-p0.10", "master_seed": 1})
        self.assertEqual(last["qec"]["distance"], 6)
        self.assertEqual(last["noise"]["parameters"]["physical_error_rate"], 0.1)
        self.assertEqual(last["model"]["parameters"]["hidden_units"], 128)
        self.assertEqual(last["dataset"]["train_samples"], 7)
        self.assertEqual(last["training"], BASE["training"])
        self.assertIsNot(points[0].config["qec"], points[1].config["qec"])
        again = qec.config_grid(
            BASE,
            name="run-l{L}-p{p:.2f}",
            axes=AXES,
            coupled=COUPLED,
            overrides={"dataset.train_samples": 7},
        )
        self.assertEqual(again, points)

    def test_invalid_declarations_fail_before_any_point_is_returned(self) -> None:
        cases = {
            "misspelled path": dict(overrides={"training.epoch": 3}),
            "section instead of field": dict(overrides={"training": {}}),
            "override conflicts with axis": dict(overrides={"qec.distance": 5}),
            "override sets the name": dict(overrides={"experiment.name": "x"}),
            "coupled conflicts with override": dict(
                coupled=COUPLED, overrides={"model.parameters.hidden_units": 1}
            ),
            "coupled lacks a value": dict(coupled={"L": {4: {"training.epochs": 1}}}),
            "coupled unknown axis": dict(coupled={"x": {}}),
            "duplicate names": dict(name="same-{L}"),
        }
        for label, arguments in cases.items():
            kwargs = {"name": "run-l{L}-p{p}", "axes": AXES, **arguments}
            with self.subTest(case=label), self.assertRaises(qec.ConfigurationError):
                qec.config_grid(BASE, **kwargs)

    def test_with_overrides_reports_the_full_missing_path(self) -> None:
        with self.assertRaises(qec.MissingConfigurationError) as caught:
            qec.with_overrides(BASE, {"noise.parameters.rate": 0.1})
        self.assertEqual(caught.exception.paths, ("noise.parameters.rate",))
        updated = qec.with_overrides(BASE, {"training.epochs": 5})
        self.assertEqual((updated["training"]["epochs"], BASE["training"]["epochs"]), (5, 30))


if __name__ == "__main__":
    unittest.main()
