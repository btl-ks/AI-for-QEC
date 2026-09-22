import unittest

import ai_qec.notebook_api as qec
from ai_qec.experiment.reuse import STAGE_INDEPENDENT_FIELDS, stage_reuse_key
from tests.helpers import tiny_config

DATASET = qec.ArtifactRef(
    "ds-1", qec.ArtifactKind.DATASET, "datasets/ds-1/manifest.json", "sha256:d", "application/json"
)
MODEL = qec.ArtifactRef(
    "attempt-0001.model-final",
    qec.ArtifactKind.MODEL_CHECKPOINT,
    "runs/a/attempts/attempt-0001/checkpoints/model/final.pt",
    "sha256:m",
    "application/x-pytorch",
)
INPUTS = {
    "training": (DATASET,),
    "scientific_evaluation": (DATASET, MODEL),
    "performance": (DATASET, MODEL),
}
# One changed value per configuration field that some Stage may ignore.
CHANGES = {
    "experiment.name": "renamed",
    "model.decoding": {"burn_in": 1, "max_steps": 2},
    "scientific_evaluation": {"baseline_decoders": ["other"]},
    "accuracy_gate": {"comparison_rule": "paired-relative-non-inferiority", "tolerance": 0.15},
    "performance": {"shots": 1, "repetitions": 1, "warmup": 0},
    "training.epochs": 4,
    "experiment.master_seed": 12,
    "execution.device": "cuda",
}


def _changed(path: str) -> dict:
    config = tiny_config()
    section, _, key = path.partition(".")
    if key:
        config[section][key] = CHANGES[path]
    else:
        config[section] = {**config[section], **CHANGES[path]}
    return config


class StageReuseKeyTests(unittest.TestCase):
    def test_only_the_declared_fields_are_ignored(self) -> None:
        for stage, inputs in INPUTS.items():
            base = stage_reuse_key(stage, tiny_config(), inputs)
            for path in CHANGES:
                with self.subTest(stage=stage, field=path):
                    changed = stage_reuse_key(stage, _changed(path), inputs)
                    if path in STAGE_INDEPENDENT_FIELDS[stage]:
                        self.assertEqual(changed, base)
                    else:
                        self.assertNotEqual(changed, base)

    def test_inputs_and_stage_enter_the_key(self) -> None:
        config = tiny_config()
        other = qec.ArtifactRef(
            "ds-2", qec.ArtifactKind.DATASET, "datasets/ds-2/manifest.json", "sha256:e", "application/json"
        )
        self.assertNotEqual(
            stage_reuse_key("training", config, (DATASET,)),
            stage_reuse_key("training", config, (other,)),
        )
        self.assertNotEqual(
            stage_reuse_key("scientific_evaluation", config, (DATASET, MODEL)),
            stage_reuse_key("performance", config, (DATASET, MODEL)),
        )

    def test_gate_and_visualization_are_never_keyed(self) -> None:
        for stage in ("dataset", "accuracy_gate", "visualization"):
            with self.subTest(stage=stage), self.assertRaises(KeyError):
                stage_reuse_key(stage, tiny_config(), ())

    def test_the_configuration_is_not_modified(self) -> None:
        config = tiny_config()
        stage_reuse_key("training", config, (DATASET,))
        self.assertEqual(config, tiny_config())


if __name__ == "__main__":
    unittest.main()
