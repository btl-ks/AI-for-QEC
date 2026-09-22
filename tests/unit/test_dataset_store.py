import dataclasses
from pathlib import Path
import tempfile
import unittest

from tests.helpers import requires_runtime, tiny_config


@requires_runtime
class DatasetStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        import ai_qec.notebook_api as qec
        from ai_qec.data.datasets.local import (
            LocalDatasetResolver,
            LocalDatasetStore,
            code_consistency_validator,
        )
        from ai_qec.data.generators.stim_code_capacity import build_stim_code_capacity_generator
        from ai_qec.experiment.config import parse_experiment_config
        from ai_qec.qec.backends.stim_noise import StimNoiseCompiler
        from ai_qec.qec.codes.toric import ToricCode

        self.qec = qec
        self.parse = parse_experiment_config
        self.root = Path(tempfile.mkdtemp())
        self.spec = parse_experiment_config(tiny_config()).spec.dataset
        self.code = ToricCode(3)
        compilation = StimNoiseCompiler().compile(self.spec.noise)
        self.store = LocalDatasetStore(self.root)

        def make_resolver(validator=None):
            return LocalDatasetResolver(
                self.store,
                generator_factory=lambda artifact_id: build_stim_code_capacity_generator(
                    code=self.code, noise=compilation, dataset_artifact_id=artifact_id
                ),
                validator=validator or code_consistency_validator(self.code),
                provenance={"code": self.code.describe()},
            )

        self.make_resolver = make_resolver

    def test_key_ignores_execution_and_training(self) -> None:
        base = self.parse(tiny_config())
        other = self.parse(
            tiny_config(
                **{
                    "execution.gpu_count": 1,
                    "execution.device": "cuda",
                    "execution.step_executor": "pytorch-cuda-graph",
                    "execution.step_executor_options": {"max_graphs": 2},
                    "training.epochs": 9,
                    "model.decoding": {"burn_in": 1, "max_steps": 2},
                }
            )
        )
        self.assertEqual(base.spec.dataset, other.spec.dataset)
        identity = self.qec.DatasetKey
        from ai_qec.data.datasets.local import CanonicalDatasetIdentity

        keys = {CanonicalDatasetIdentity().key_for(item.spec.dataset) for item in (base, other)}
        self.assertEqual(len(keys), 1)
        self.assertIsInstance(next(iter(keys)), identity)

    def test_miss_commits_then_hit_reuses_the_same_verified_artifact(self) -> None:
        first = self.make_resolver().resolve(self.spec, "attempt-0001")
        second = self.make_resolver().resolve(self.spec, "attempt-0002")
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.artifact, second.artifact)
        self.assertNotEqual(first.instance.instance_id, second.instance.instance_id)
        self.assertTrue(self.store.verify(first.artifact))
        self.assertEqual([split.sample_count for split in first.artifact.splits], [1000, 100, 200])
        test = self.store.load_split(first.artifact, "test")
        self.assertEqual(test.physical_errors.shape, (200, 18))
        self.assertEqual(test.context["persisted_representation"], "bit-packed-cpu-buffer")

    def test_corrupt_shard_rejects_cache_hit_and_reads(self) -> None:
        from ai_qec.data.datasets.local import DatasetIntegrityError

        artifact = self.make_resolver().resolve(self.spec, "attempt-0001").artifact
        shard = self.store.path(artifact.splits[2].shard_uris[0])
        data = bytearray(shard.read_bytes())
        data[-5] ^= 0xFF
        shard.write_bytes(bytes(data))
        with self.assertRaisesRegex(DatasetIntegrityError, "checksum mismatch"):
            self.make_resolver().resolve(self.spec, "attempt-0002")
        with self.assertRaises(DatasetIntegrityError):
            self.store.load_split(artifact, "test")
        self.store.load_split(artifact, "train")

    def test_inconsistent_samples_are_never_committed(self) -> None:
        from ai_qec.data.datasets.local import DatasetIntegrityError

        def reject_test_split(batch) -> None:
            if batch.context["split"] == "test":
                raise DatasetIntegrityError("[test] injected inconsistency")

        with self.assertRaisesRegex(DatasetIntegrityError, "injected"):
            self.make_resolver(reject_test_split).resolve(self.spec, "attempt-0001")
        self.assertFalse(self.store.registry_path.exists())
        self.assertEqual(
            [path.name for path in self.root.iterdir() if path.name.startswith("ds-")], []
        )

    def test_validator_detects_syndrome_mismatch(self) -> None:
        from ai_qec.data.datasets.local import DatasetIntegrityError, code_consistency_validator

        artifact = self.make_resolver().resolve(self.spec, "attempt-0001").artifact
        batch = self.store.load_split(artifact, "validation")
        corrupted = batch.detector_events.copy()
        corrupted[7, 0] ^= 1
        with self.assertRaisesRegex(DatasetIntegrityError, "detector_events != S"):
            code_consistency_validator(self.code)(
                dataclasses.replace(batch, detector_events=corrupted)
            )


if __name__ == "__main__":
    unittest.main()
