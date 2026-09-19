"""Phase-0 contract, data-integrity, geometry, and fail-loudly tests."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_qec.data.datasets.qec_dataset import validate_dataset  # noqa: E402
from ai_qec.data.generators.qec_generator import generate_dataset  # noqa: E402
from ai_qec.data.generators.qec_generator import _validate_batch  # noqa: E402
from ai_qec.qec.codes.surface_code import SurfaceCode  # noqa: E402
from ai_qec.qec.circuits.memory import MemoryExperiment  # noqa: E402
from ai_qec.utils.config import ConfigValidationError, data_output_dir, load_config, resolve_experiment_spec  # noqa: E402
from ai_qec.utils.reproducibility import interpreter_environment  # noqa: E402


def config_dict(tmp: Path) -> dict:
    data = yaml.safe_load((PROJECT_ROOT / "configs" / "experiment.smoke.yaml").read_text(encoding="utf-8"))
    data.pop("paper_export")
    data["data"].update(train_samples=12, validation_samples=6, test_samples=6, output_dir=str(tmp / "datasets"), dataset_id="unit")
    return data


class ConfigAndDatasetContractTest(unittest.TestCase):
    def test_unknown_unavailable_and_multiseed_are_rejected_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            config = config_dict(Path(tmp_name))
            bad = copy.deepcopy(config); bad["unknown"] = True
            with self.assertRaises(ConfigValidationError): resolve_experiment_spec(bad, PROJECT_ROOT)
            bad = copy.deepcopy(config); bad["data"]["generator"] = "stim"
            with self.assertRaisesRegex(ConfigValidationError, "not implemented"): resolve_experiment_spec(bad, PROJECT_ROOT)
            bad = copy.deepcopy(config); bad["model"]["implementation"] = "transformer_decoder"
            with self.assertRaisesRegex(ConfigValidationError, "not implemented"): resolve_experiment_spec(bad, PROJECT_ROOT)
            bad = copy.deepcopy(config); bad["reproducibility"]["seeds"] = [1, 2]
            with self.assertRaisesRegex(ConfigValidationError, "multiple seeds"): resolve_experiment_spec(bad, PROJECT_ROOT)

    def test_optional_conda_environment_name_is_strictly_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            config = config_dict(Path(tmp_name))
            config["execution"] = {"conda_env": "quantum"}
            self.assertEqual(resolve_experiment_spec(config, PROJECT_ROOT).config["execution"]["conda_env"], "quantum")

            invalid = copy.deepcopy(config); invalid["execution"] = {"conda_env": "../../other"}
            with self.assertRaisesRegex(ConfigValidationError, "Conda environment name"):
                resolve_experiment_spec(invalid, PROJECT_ROOT)
            invalid = copy.deepcopy(config); invalid["execution"] = {"conda_env": "quantum", "python": "python"}
            with self.assertRaisesRegex(ConfigValidationError, "unknown field"):
                resolve_experiment_spec(invalid, PROJECT_ROOT)

    def test_environment_snapshot_uses_the_requested_interpreter(self) -> None:
        snapshot = interpreter_environment(sys.executable)
        self.assertEqual(Path(snapshot["executable"]).resolve(), Path(sys.executable).resolve())
        self.assertIn("numpy", snapshot["packages"])
        self.assertIn("numpy", {name.lower() for name in snapshot["distributions"]})

    def test_immutable_dataset_rejects_tampering_and_stale_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            raw = config_dict(tmp)
            path = tmp / "config.yaml"; path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            config = load_config(path, PROJECT_ROOT)
            generate_dataset(config, PROJECT_ROOT)
            root = data_output_dir(config, PROJECT_ROOT)
            self.assertTrue(validate_dataset(root, config))
            self.assertTrue(generate_dataset(config, PROJECT_ROOT)["reused_immutable"])
            (root / "test.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"): validate_dataset(root, config)

    def test_backend_batch_missing_field_fails_before_write(self) -> None:
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            _validate_batch({}, 1, 8)

    def test_rotated_surface_geometry_is_not_detector_geometry(self) -> None:
        for distance in (3, 5, 7):
            code = SurfaceCode(distance=distance)
            self.assertEqual(code.num_data_qubits, distance ** 2)
            self.assertEqual(code.num_stabilizers_per_round, distance ** 2 - 1)
            self.assertEqual(code.num_measurement_qubits, distance ** 2 - 1)
            for rounds in (1, 3):
                self.assertEqual(MemoryExperiment(code, rounds).num_detectors, rounds * (distance ** 2 - 1))

    def test_export_path_guard_rejects_project_root_and_existing_release(self) -> None:
        spec = importlib.util.spec_from_file_location("export_paper", PROJECT_ROOT / "scripts" / "export_paper.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name); (root / "paper" / "releases").mkdir(parents=True)
            with self.assertRaises(module.ExportError):
                module.resolve_package_root(root, root, "x", {})
            package = module.resolve_package_root(None, root, "x", {})
            package.mkdir()
            (package / "same").mkdir()
            with self.assertRaises(module.ExportError):
                module.publish_release(package, "same", lambda _: None)
            outside = root / "outside"; outside.mkdir()
            (root / "paper" / "releases" / "escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(module.ExportError):
                module.resolve_package_root(root / "paper" / "releases" / "escape", root, "escape", {})


# Run the command-line entry point when this module is executed directly.
if __name__ == "__main__":
    unittest.main()
