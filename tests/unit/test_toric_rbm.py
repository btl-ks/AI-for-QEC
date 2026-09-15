"""Physics and integration contracts for the Torlai--Melko reproduction path."""

from __future__ import annotations

import json
import importlib.util
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
from ai_qec.utils.config import load_config  # noqa: E402


class ToricRBMContractTest(unittest.TestCase):
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
            raw = yaml.safe_load((PROJECT_ROOT / "configs" / "experiment.torlai_melko_2017.smoke.yaml").read_text(encoding="utf-8"))
            raw["data"].update(train_samples=64, validation_samples=16, test_samples=8, batch_size=32, output_dir=str(tmp / "datasets"), dataset_id="toric-unit")
            raw["training"].update(epochs=2, batch_size=32, decoder={"burn_in": 2, "max_steps": 32, "parallel_chains": 8})
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
