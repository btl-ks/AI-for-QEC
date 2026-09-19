"""Shared experiment setup contracts used by notebook entry points."""

from __future__ import annotations

import unittest
from pathlib import Path

from ai_qec.utils.config import require_experiment_kind, resolve_notebook_config
from ai_qec.utils.experiment_setup import find_project_root, prepare_experiment


class ExperimentSetupTest(unittest.TestCase):
    def test_finds_source_root_outside_project_directory(self) -> None:
        root = find_project_root(Path("/tmp"))
        self.assertTrue((root / "ai_qec" / "__init__.py").is_file())
        self.assertTrue((root / "paper" / "srcs" / "torlai_melko_2017.ipynb").is_file())

    def test_cpu_setup_builds_code_noise_and_seed(self) -> None:
        config = {
            "training": {"device": "cpu"},
            "reproducibility": {"seeds": [17]},
            "qec": {"code": "toric_code", "distance": 4},
            "noise": {"model": "phase_flip", "p_error": 0.08},
        }
        setup = prepare_experiment(config)
        self.assertEqual((setup.device, setup.seed), ("cpu", 17))
        self.assertEqual((setup.code.num_data_qubits, setup.code.num_syndrome_bits), (32, 16))
        self.assertEqual(setup.noise.p_error, 0.08)

    def test_wrong_environment_fails_before_building_objects(self) -> None:
        config = {"execution": {"conda_env": "not_the_active_environment"}}
        with self.assertRaisesRegex(RuntimeError, "not_the_active_environment"):
            prepare_experiment(config)

    def test_notebook_config_rejects_duplicate_keys_before_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "qec"):
            resolve_notebook_config(
                {"qec": {"code": "toric_code"}}, {"qec": {"code": "surface_code"}}, project_root=find_project_root()
            )

    def test_experiment_kind_check_uses_only_experiment_parameters(self) -> None:
        parameters = {"data": {"generator": "toric_code_capacity"}, "model": {"implementation": "joint_error_syndrome_rbm"}}
        require_experiment_kind(parameters, generator="toric_code_capacity", model="joint_error_syndrome_rbm")
        with self.assertRaisesRegex(ValueError, "model.implementation"):
            require_experiment_kind(parameters, generator="toric_code_capacity", model="other_model")


# Run the command-line entry point when this module is executed directly.
if __name__ == "__main__":
    unittest.main()
