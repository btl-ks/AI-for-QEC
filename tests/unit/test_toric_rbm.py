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

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch dependency is absent")
    def test_validated_unsynced_cd_step_matches_default_step_exactly(self) -> None:
        import torch

        from ai_qec.models.decoders.generative.rbm import JointErrorSyndromeRBM

        rng = np.random.default_rng(5)
        data = (rng.random((256, 12)) < 0.2).astype(np.uint8)
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                runs = []
                for fast in (False, True):
                    model = JointErrorSyndromeRBM(error_units=8, syndrome_units=4, hidden_units=6, init_width=0.1, seed=3).to(device)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4)
                    generator = torch.Generator(device=device).manual_seed(11)
                    split = model.to_visible_tensor(data) if fast else data
                    losses = [
                        model.contrastive_divergence_step(
                            split[start:start + 32], optimizer=optimizer, cd_steps=2, generator=generator,
                            validated=fast, sync=not fast,
                        )
                        for start in range(0, len(data), 32)
                    ]
                    if fast:
                        losses = torch.stack(losses).cpu().numpy().astype(np.float64).tolist()
                    runs.append((losses, [p.detach().clone() for p in model.parameters()], generator.get_state()))
                (slow_losses, slow_params, slow_rng), (fast_losses, fast_params, fast_rng) = runs
                self.assertEqual(slow_losses, fast_losses)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(slow_params, fast_params)))
                self.assertTrue(torch.equal(slow_rng, fast_rng))
                with self.assertRaises(ValueError):
                    model.contrastive_divergence_step(data[:4], optimizer=optimizer, cd_steps=1, generator=generator, validated=True)

    @unittest.skipUnless(importlib.util.find_spec("pymatching") is not None, "optional PyMatching dependency is absent")
    def test_mwpm_reference_uses_pymatching_only_above_exact_defect_limit(self) -> None:
        from ai_qec.benchmarks.decoding.toric import mwpm_reference_recoveries
        from ai_qec.models.decoders.classical.pymatching_adapter import PyMatchingToricDecoder

        code = ToricCode(distance=6)
        rng = np.random.default_rng(9)
        errors = (rng.random((400, code.num_data_qubits)) < 0.15).astype(np.uint8)
        syndromes = code.syndrome(errors)
        large = syndromes.sum(axis=1) > ExactToricMWPMDecoder(code).max_exact_defects
        self.assertTrue(large.any() and not large.all())
        recoveries, limit, fallback = mwpm_reference_recoveries(code, syndromes)
        self.assertEqual((limit, fallback), (ExactToricMWPMDecoder(code).max_exact_defects, int(large.sum())))
        self.assertTrue(np.array_equal(code.syndrome(recoveries), syndromes))
        pymatching = PyMatchingToricDecoder(code)
        for index in np.flatnonzero(large)[:5]:
            self.assertEqual(recoveries[index].sum(), pymatching.decode(DecodeRequest(syndromes[index])).recovery.sum())
        exact = ExactToricMWPMDecoder(code)
        for index in np.flatnonzero(~large)[:5]:
            self.assertTrue(np.array_equal(recoveries[index], exact.decode(syndromes[index])))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch dependency is absent")
    def test_reused_checkpoint_run_skips_training_and_reproduces_source_decoding(self) -> None:
        import copy
        import torch

        import ai_qec.notebook_api as qec
        from ai_qec.models.decoders.generative.rbm_decoder import RBMGibbsDecoder

        run_keys = {"schema_version", "execution", "experiment", "topic", "reproducibility", "outputs"}

        def start(raw: dict):
            run_config = {k: v for k, v in raw.items() if k in run_keys}
            experiment = {k: v for k, v in raw.items() if k not in run_keys}
            with redirect_stdout(io.StringIO()):
                return qec.start_notebook_run(
                    run_config, experiment, project_root=PROJECT_ROOT, notebook="paper/srcs/torlai_melko_2017.ipynb",
                    prepared_experiment_hash=qec.config_hash(experiment),
                )[0]

        def decode(record, checkpoint: Path, test) -> dict:
            model, _ = qec.load_model(record.config, str(checkpoint))
            settings = record.config["training"]["decoder"]
            decoder = RBMGibbsDecoder(
                model, code, burn_in=settings["burn_in"], max_steps=settings["max_steps"],
                parallel_chains=settings["parallel_chains"], device="cpu",
            )
            out = {"recovery": np.zeros_like(test.physical_error), "valid": np.zeros(len(test.syndrome), np.uint8),
                   "steps": np.zeros(len(test.syndrome), np.int32)}
            for index, syndrome in enumerate(test.syndrome):
                result = decoder.decode(syndrome, rng=qec.decoding_rng(record.config, index))
                out["steps"][index] = result.steps
                if result.success:
                    out["recovery"][index], out["valid"][index] = result.recovery, 1
            return out

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            raw = notebook_raw_config()
            raw["outputs"]["runs_root"] = str(tmp / "runs")
            raw["data"].update(train_samples=64, validation_samples=16, test_samples=12, batch_size=32,
                               output_dir=str(tmp / "datasets"), dataset_id="reuse-unit")
            raw["model"]["hidden_units"] = 6
            raw["training"].update(device="cpu", epochs=1, batch_size=32,
                                   decoder={"burn_in": 1, "max_steps": 150, "parallel_chains": 64})
            raw["reproducibility"].update(save_environment=False, save_git_commit=False)
            raw["execution"]["conda_env"] = Path(sys.prefix).name

            source = start(raw)
            config = source.config
            code, noise = qec.build_code(config), qec.build_noise_model(config)
            dataset_dir = qec.data_output_dir(config, PROJECT_ROOT)
            with source.stage("generate_data") as step:
                with qec.staged_dataset_dir(dataset_dir) as staging:
                    files = {}
                    for split, count in qec.split_sample_counts(config).items():
                        errors = noise.sample_errors(count, code.num_data_qubits, np.random.default_rng(17 + qec.SPLIT_SEED_OFFSETS[split]))
                        files[f"{split}.npz"] = qec.write_toric_split(
                            staging / f"{split}.npz", split=split, dataset_id="reuse-unit", physical_error=errors,
                            syndrome=code.syndrome(errors), lattice_size=code.distance, p_error=noise.p_error,
                        )
                    qec.write_json(staging / "dataset_manifest.json", qec.build_toric_dataset_manifest(config, code, files))
                dataset_manifest = qec.validate_toric_dataset(dataset_dir, config)
                step["outputs"].append(dataset_dir / "dataset_manifest.json")
            source.refresh_dataset()
            splits = {name: qec.load_toric_split(dataset_dir, name) for name in ("train", "validation", "test")}
            best = source.run_dir / "checkpoints" / "best.pt"
            with source.stage("train") as step:
                model = qec.build_model(config, error_units=code.num_data_qubits, syndrome_units=code.num_syndrome_bits)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
                model.contrastive_divergence_step(splits["train"].visible[:32], optimizer=optimizer, cd_steps=1, generator=torch.Generator().manual_seed(1))
                row = {"epoch": 1, "train_reconstruction_bce": 0.5, "validation_reconstruction_bce": model.reconstruction_bce(splits["validation"].visible)}
                model.save(best, optimizer=optimizer, metadata=qec.rbm_checkpoint_metadata(config, dataset_manifest, PROJECT_ROOT, metrics=row, selection="unit"))
                summary_path = source.run_dir / "training_summary.json"
                qec.write_json(summary_path, qec.rbm_training_summary([row], selected_epoch=1, training_time_seconds=0.0, dataset_dir=dataset_dir))
                step["outputs"] += [best, summary_path]
            first = decode(source, best, splits["test"])
            with source.stage("evaluate") as step:
                predictions = source.run_dir / "predictions" / "toric_rbm_eval.npz"
                failures = np.ones(len(first["valid"]), np.uint8)
                qec.save_toric_predictions(
                    predictions, dataset=splits["test"], parallel_chains=64, device="cpu", recovery=first["recovery"],
                    recovery_valid=first["valid"], timed_out=1 - first["valid"], gibbs_steps=first["steps"],
                    decoder_latency_ms=np.zeros(len(first["valid"])), logical_failure=failures,
                )
                step["outputs"].append(predictions)
            source.finish()

            reuse_raw = copy.deepcopy(raw)
            reuse_raw["experiment"]["name"] = "reuse-unit-redecode"
            reuse_raw["training"]["decoder"]["max_steps"] = 600
            reuse = start(reuse_raw)
            loaded = qec.load_source_run(source.run_id, runs_root=tmp / "runs")
            with reuse.stage("generate_data") as step:
                reused_dir, reused_manifest = qec.validate_source_dataset(loaded, reuse.config)
                step["outputs"].append(reused_dir / "dataset_manifest.json")
            qec.link_source_dataset(reuse, loaded)
            self.assertEqual(reused_dir, dataset_dir)
            self.assertEqual(reuse.manifest["dataset"]["validated_against_source_run"], source.run_id)
            self.assertNotEqual(qec.data_output_dir(reuse.config, PROJECT_ROOT), dataset_dir)

            copied = qec.copy_source_checkpoint(loaded, reuse.run_dir / "checkpoints" / "best.pt",
                                                config=reuse.config, dataset_manifest=reused_manifest)
            self.assertEqual(copied.read_bytes(), best.read_bytes())
            self.assertEqual(qec.source_training_summary(loaded)["metrics"]["selected_epoch"], 1)
            second = decode(reuse, copied, splits["test"])
            report = qec.compare_with_source_predictions(
                loaded, config=reuse.config, syndromes=splits["test"].syndrome, recovery_valid=second["valid"],
                gibbs_steps=second["steps"], recovery=second["recovery"],
            )
            self.assertTrue(report["checked"])
            self.assertTrue(report["identical"])
            self.assertEqual(report["shared_step_budget"], 150)
            self.assertGreater(report["shots_resolved_within_budget"], 0)
            self.assertGreaterEqual(report["valid"], report["source_valid"])

            tampered = second["steps"].copy()
            tampered[np.flatnonzero((first["valid"] == 1))[0]] += 1
            self.assertFalse(qec.compare_with_source_predictions(
                loaded, config=reuse.config, syndromes=splits["test"].syndrome, recovery_valid=second["valid"],
                gibbs_steps=tampered, recovery=second["recovery"],
            )["identical"])

            changed = copy.deepcopy(reuse.config)
            changed["model"]["hidden_units"] = 7
            with self.assertRaisesRegex(ValueError, "model"):
                qec.validate_source_dataset(loaded, changed)
            changed = copy.deepcopy(reuse.config)
            changed["training"]["epochs"] = 2
            with self.assertRaisesRegex(ValueError, "training"):
                qec.validate_source_dataset(loaded, changed)

            with best.open("ab") as handle:
                handle.write(b"x")
            with self.assertRaisesRegex(RuntimeError, "recorded hash"):
                qec.copy_source_checkpoint(loaded, tmp / "other.pt", config=reuse.config, dataset_manifest=reused_manifest)

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
