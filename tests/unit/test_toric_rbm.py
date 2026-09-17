"""Physics and integration contracts for the Torlai--Melko reproduction path."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import json
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "run_experiment.py"
sys.path.insert(0, str(PROJECT_ROOT))

from ai_qec.models.decoders.classical.mwpm import ExactToricMWPMDecoder  # noqa: E402
from ai_qec.models.decoders.protocol import DecodeRequest  # noqa: E402
from ai_qec.qec.codes.toric_code import ToricCode  # noqa: E402
from ai_qec.utils.config import config_hash, load_config  # noqa: E402
from ai_qec.utils.run_record import start_notebook_run  # noqa: E402


def notebook_raw_config() -> dict:
    notebook = json.loads((PROJECT_ROOT / "paper" / "srcs" / "torlai_melko_2017.ipynb").read_text(encoding="utf-8"))
    configs = {}
    for index in (3, 5):
        config_cell = ast.parse("".join(notebook["cells"][index]["source"]))
        for node in config_cell.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"RUN_CONFIG", "EXPERIMENT_CONFIG"}:
                        configs[target.id] = ast.literal_eval(node.value)
    if set(configs) != {"RUN_CONFIG", "EXPERIMENT_CONFIG"}:
        raise AssertionError("Notebook must define literal RUN_CONFIG and EXPERIMENT_CONFIG")
    if set(configs["RUN_CONFIG"]) & set(configs["EXPERIMENT_CONFIG"]):
        raise AssertionError("Notebook config groups must not overlap")
    return {**configs["RUN_CONFIG"], **configs["EXPERIMENT_CONFIG"]}


class ToricRBMContractTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch dependency is absent")
    def test_first_compatible_chain_uses_exact_toric_parity_on_available_devices(self) -> None:
        import torch

        from ai_qec.models.decoders.generative.rbm_decoder import first_compatible_chain

        code = ToricCode(distance=4)
        errors = np.zeros((3, code.num_data_qubits), dtype=np.uint8)
        errors[1, 0] = 1
        errors[2, 1] = 1
        target_np = code.syndrome(errors[1])
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                chains = torch.as_tensor(errors, dtype=torch.float32, device=device)
                target = torch.as_tensor(target_np, dtype=torch.int64, device=device)
                checks = torch.as_tensor(code.parity_check_matrix(), dtype=torch.int64, device=device)
                self.assertEqual(first_compatible_chain(chains, target, checks), 1)
                self.assertIsNone(first_compatible_chain(chains[:1], target, checks))

    def test_notebook_inline_config_is_validated_and_snapshotted(self) -> None:
        raw = notebook_raw_config()
        self.assertNotIn("flow", raw)
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            raw["outputs"]["runs_root"] = str(tmp / "runs")
            raw["data"]["output_dir"] = str(tmp / "datasets")
            raw["reproducibility"].update(save_environment=False, save_git_commit=False)
            raw["execution"]["conda_env"] = Path(sys.prefix).name
            raw["training"]["device"] = "cpu"
            run_keys = {"schema_version", "execution", "experiment", "topic", "reproducibility", "outputs"}
            run_config = {key: value for key, value in raw.items() if key in run_keys}
            experiment_config = {key: value for key, value in raw.items() if key not in run_keys}
            with self.assertRaisesRegex(RuntimeError, "实验参数已更改"):
                start_notebook_run(
                    run_config, experiment_config, project_root=PROJECT_ROOT,
                    notebook="paper/srcs/torlai_melko_2017.ipynb", prepared_experiment_hash="stale",
                )
            self.assertFalse((tmp / "runs").exists())
            output = io.StringIO()
            with redirect_stdout(output):
                record, device, seed = start_notebook_run(
                    run_config, experiment_config, project_root=PROJECT_ROOT,
                    notebook="paper/srcs/torlai_melko_2017.ipynb",
                    prepared_experiment_hash=config_hash(experiment_config),
                )
            config = record.config
            self.assertEqual((device, seed), ("cpu", 17))
            self.assertIn("schema_version:", output.getvalue())
            self.assertIn("qec:", output.getvalue())
            self.assertIn("outputs:", output.getvalue())
            self.assertLess(output.getvalue().index("outputs:"), output.getvalue().index("run 已创建："))
            snapshot = yaml.safe_load((record.run_dir / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(snapshot, config)
            self.assertEqual(record.manifest["config_hash"], config_hash(snapshot))
            self.assertEqual(record.manifest["config"], "paper/srcs/torlai_melko_2017.ipynb")
            self.assertIn("run started", (record.run_dir / "run.log").read_text(encoding="utf-8"))

    def test_toric_geometry_has_correct_boundaries_and_winding_classes(self) -> None:
        """Check individual edges, a plaquette boundary, and both logical loops."""
        code = ToricCode(distance=4)
        l = code.distance
        horizontal = np.zeros((l, l), dtype=np.uint8)
        vertical = np.zeros((l, l), dtype=np.uint8)

        horizontal[1, 2] = 1
        one_edge = np.concatenate((horizontal.ravel(), vertical.ravel()))
        expected = np.zeros((l, l), dtype=np.uint8)
        expected[1, 2] = expected[1, 3] = 1
        self.assertTrue(np.array_equal(code.syndrome(one_edge).reshape(l, l), expected))

        horizontal.fill(0)
        vertical[3, 1] = 1  # crosses the periodic y-boundary
        one_edge = np.concatenate((horizontal.ravel(), vertical.ravel()))
        expected.fill(0)
        expected[3, 1] = expected[0, 1] = 1
        self.assertTrue(np.array_equal(code.syndrome(one_edge).reshape(l, l), expected))

        horizontal.fill(0)
        vertical.fill(0)
        horizontal[1, 2] = horizontal[2, 2] = 1
        vertical[1, 2] = vertical[1, 3] = 1
        plaquette = np.concatenate((horizontal.ravel(), vertical.ravel()))
        self.assertFalse(np.any(code.syndrome(plaquette)))
        self.assertEqual(code.homology(plaquette).tolist(), [0, 0])

        horizontal.fill(0)
        vertical.fill(0)
        horizontal[1, :] = 1
        horizontal_loop = np.concatenate((horizontal.ravel(), vertical.ravel()))
        self.assertFalse(np.any(code.syndrome(horizontal_loop)))
        self.assertEqual(code.homology(horizontal_loop).tolist(), [1, 0])

        horizontal.fill(0)
        vertical[:, 2] = 1
        vertical_loop = np.concatenate((horizontal.ravel(), vertical.ravel()))
        self.assertFalse(np.any(code.syndrome(vertical_loop)))
        self.assertEqual(code.homology(vertical_loop).tolist(), [0, 1])

    def test_l6_geometry_matches_paper_scale_code_capacity_layout(self) -> None:
        code = ToricCode(distance=6)
        self.assertEqual(code.num_data_qubits, 72)
        self.assertEqual(code.num_syndrome_bits, 36)
        horizontal = np.zeros((6, 6), dtype=np.uint8)
        vertical = np.zeros((6, 6), dtype=np.uint8)
        horizontal[4, :] = 1
        vertical[:, 3] = 1
        both_loops = np.concatenate((horizontal.ravel(), vertical.ravel()))
        self.assertFalse(np.any(code.syndrome(both_loops)))
        self.assertEqual(code.homology(both_loops).tolist(), [1, 1])

    @unittest.skipUnless(importlib.util.find_spec("pymatching") is not None, "optional PyMatching dependency is absent")
    def test_pymatching_adapter_agrees_on_minimum_recovery_weight(self) -> None:
        from ai_qec.models.decoders.classical.pymatching_adapter import PyMatchingToricDecoder

        code = ToricCode(distance=4)
        exact = ExactToricMWPMDecoder(code)
        matching = PyMatchingToricDecoder(code)
        rng = np.random.default_rng(91)
        errors = (rng.random((64, code.num_data_qubits)) < 0.08).astype(np.uint8)
        for syndrome in code.syndrome(errors):
            exact_result = exact.decode(DecodeRequest(syndrome))
            matching_result = matching.decode(DecodeRequest(syndrome))
            self.assertTrue(np.array_equal(code.syndrome(matching_result.recovery), syndrome))
            self.assertEqual(int(exact_result.recovery.sum()), int(matching_result.recovery.sum()))

    def test_syndrome_and_exact_matching_recovery_are_consistent(self) -> None:
        code = ToricCode(distance=4)
        rng = np.random.default_rng(23)
        errors = (rng.random((48, code.num_data_qubits)) < 0.1).astype(np.uint8)
        syndromes = code.syndrome(errors)
        self.assertTrue(np.all(np.sum(syndromes, axis=1) % 2 == 0))
        decoder = ExactToricMWPMDecoder(code)
        recoveries = np.stack([decoder.decode(syndrome) for syndrome in syndromes])
        self.assertTrue(np.array_equal(code.syndrome(recoveries), syndromes))
        result = decoder.decode(DecodeRequest(syndromes[0], 0.1))
        self.assertTrue(result.success)
        self.assertTrue(np.array_equal(code.syndrome(result.recovery), syndromes[0]))

    def test_toric_config_runs_the_traceable_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            raw = notebook_raw_config()
            raw["data"].update(train_samples=64, validation_samples=16, test_samples=8, batch_size=32, output_dir=str(tmp / "datasets"), dataset_id="toric-unit")
            raw["training"].update(epochs=2, batch_size=32, decoder={"burn_in": 2, "max_steps": 32, "parallel_chains": 8})
            raw["training"]["device"] = "cpu"
            raw["flow"] = [
                {
                    "id": "generate_data", "script": "scripts/generate_data.py",
                    "args": ["--config", "${CONFIG}", "--project-root", "${PROJECT_ROOT}"],
                    "outputs": [{"path": "${DATASET_DIR}/dataset_manifest.json", "artifact_type": "dataset_manifest", "schema_version": 1}],
                },
                {
                    "id": "train", "script": "scripts/train.py",
                    "args": ["--config", "${CONFIG}", "--run-dir", "${RUN_DIR}", "--project-root", "${PROJECT_ROOT}"],
                    "outputs": [
                        {"path": "${RUN_DIR}/checkpoints/best.pt", "artifact_type": "checkpoint"},
                        {"path": "${RUN_DIR}/checkpoints/last.pt", "artifact_type": "checkpoint"},
                        {"path": "${RUN_DIR}/training_summary.json", "artifact_type": "json"},
                    ],
                },
                {
                    "id": "evaluate", "script": "scripts/evaluate.py",
                    "args": ["--config", "${CONFIG}", "--run-dir", "${RUN_DIR}", "--checkpoint", "${RUN_DIR}/checkpoints/best.pt", "--project-root", "${PROJECT_ROOT}"],
                    "outputs": [
                        {"path": "${RUN_DIR}/predictions/toric_rbm_eval.npz", "artifact_type": "predictions"},
                        {"path": "${RUN_DIR}/metrics.json", "artifact_type": "json"},
                    ],
                },
                {
                    "id": "benchmark", "script": "scripts/benchmark.py",
                    "args": ["--config", "${CONFIG}", "--run-dir", "${RUN_DIR}", "--project-root", "${PROJECT_ROOT}"],
                    "outputs": [
                        {"path": "${RUN_DIR}/metrics.json", "artifact_type": "json"},
                        {"path": "${RUN_DIR}/benchmark_report.json", "artifact_type": "json"},
                    ],
                },
            ]
            config_path = tmp / "config.yaml"
            config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            config = load_config(config_path, PROJECT_ROOT)
            self.assertEqual(config["data"]["generator"], "toric_code_capacity")
            run_dir = tmp / "run"
            proc = subprocess.run(
                [sys.executable, str(RUNNER), "--config", str(config_path), "--project-root", str(PROJECT_ROOT), "--run-dir", str(run_dir), "--allow-dirty"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(manifest["physics_fidelity"], "toric_code_capacity")
            self.assertTrue(all(step["status"] == "success" for step in manifest["steps"]))
            self.assertTrue((run_dir / "predictions" / "toric_rbm_eval.npz").is_file())
            self.assertTrue((run_dir / "checkpoints" / "best.pt").is_file())
            report = json.loads((run_dir / "benchmark_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["samples"], 8)
            self.assertIn("logical_error_rate", report["mwpm_exact"])
            self.assertIn("decoder_latency_p95_ms", report["rbm"])


if __name__ == "__main__":
    unittest.main()
