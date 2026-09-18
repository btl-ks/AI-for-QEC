"""Shared toric sampling and PyTorch RBM training contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ai_qec.data.datasets.dataset_resolution import resolve_dataset
from ai_qec.data.datasets.toric_dataset import SPLIT_SEED_OFFSETS, load_toric_split
from ai_qec.qec.codes.registry import build_code
from ai_qec.qec.noise.registry import build_noise_model
from ai_qec.utils.config import data_output_dir, split_sample_counts
from tests.unit.test_toric_rbm import notebook_raw_config


class _RunForDataset:
    def __init__(self) -> None:
        self.refreshes = 0

    def refresh_dataset(self) -> None:
        self.refreshes += 1


def small_config(root: Path) -> dict:
    config = notebook_raw_config()
    config["data"].update(
        train_samples=11, validation_samples=5, test_samples=6,
        batch_size=4, output_dir=str(root / "datasets"), dataset_id="rbm-data-training-test",
    )
    config["training"].update(device="cpu", epochs=2, batch_size=4)
    config["model"]["hidden_units"] = 4
    return config


class RBMDataTrainingTest(unittest.TestCase):
    def test_notebook_dataset_resolution_uses_shared_split_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            config = small_config(root)
            run = _RunForDataset()
            step = {"outputs": []}
            dataset_dir, manifest = resolve_dataset(run, config, root, step=step)
            code, noise = build_code(config), build_noise_model(config)
            seed = config["reproducibility"]["seeds"][0]

            self.assertEqual(dataset_dir, data_output_dir(config, root))
            self.assertEqual(manifest["sample_counts"], split_sample_counts(config))
            self.assertEqual(run.refreshes, 1)
            for split, count in split_sample_counts(config).items():
                dataset = load_toric_split(dataset_dir, split)
                expected = noise.sample_errors(
                    count, code.num_data_qubits,
                    np.random.default_rng(seed + SPLIT_SEED_OFFSETS[split]),
                )
                self.assertEqual(dataset.visible.dtype, np.float32)
                np.testing.assert_array_equal(dataset.physical_error, expected)
                np.testing.assert_array_equal(dataset.syndrome, code.syndrome(expected))

            reused = {"outputs": []}
            resolve_dataset(run, config, root, step=reused)
            self.assertTrue(reused["reused_immutable"])

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch dependency is absent")
    def test_shared_training_is_reproducible_with_partial_final_batch(self) -> None:
        import torch

        import ai_qec.notebook_api as qec
        from ai_qec.data.generators.toric_generator import generate_toric_dataset
        from ai_qec.training.trainers.rbm import train_rbm

        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            config = small_config(root)
            generate_toric_dataset(config, root)

            first = qec.run_rbm_training(config, root, root / "first")
            second_metrics = train_rbm(config, root, root / "second")
            first_metrics = {key: value for key, value in first.summary["metrics"].items() if key != "training_time_seconds"}
            second_metrics = {key: value for key, value in second_metrics.items() if key != "training_time_seconds"}
            self.assertEqual(first_metrics, second_metrics)
            self.assertEqual(len(first.summary["history"]), 2)
            self.assertEqual(len(first.artifacts), 3)
            self.assertTrue(all(path.is_file() for path in first.artifacts))

            first_state = torch.load(root / "first" / "checkpoints" / "last.pt", weights_only=True)["model_state"]
            second_state = torch.load(root / "second" / "checkpoints" / "last.pt", weights_only=True)["model_state"]
            self.assertTrue(all(torch.equal(first_state[key], second_state[key]) for key in first_state))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch dependency is absent")
    def test_training_metric_weights_the_partial_batch_by_sample_count(self) -> None:
        import torch

        from ai_qec.data.generators.toric_generator import generate_toric_dataset
        from ai_qec.models.decoders.generative.rbm import JointErrorSyndromeRBM
        from ai_qec.training.trainers.rbm import run_rbm_training

        def sized_loss(model, visible, *, optimizer, cd_steps, generator, validated=False, sync=True):
            self.assertIsInstance(model, JointErrorSyndromeRBM)
            self.assertTrue(validated)
            self.assertFalse(sync)
            return torch.tensor(float(len(visible)), device=visible.device)

        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            config = small_config(root)
            config["training"]["epochs"] = 1
            generate_toric_dataset(config, root)
            with patch.object(JointErrorSyndromeRBM, "contrastive_divergence_step", sized_loss):
                result = run_rbm_training(config, root, root / "run")
            self.assertAlmostEqual(result.summary["history"][0]["train_reconstruction_bce"], 41 / 11, places=5)


if __name__ == "__main__":
    unittest.main()
