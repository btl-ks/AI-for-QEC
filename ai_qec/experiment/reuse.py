"""Stage reuse keys for sharing completed Stages across Experiments.

A key covers the Stage name, its input ArtifactRefs and the full canonical
configuration minus the fields that neither the Stage nor any upstream Stage
reads. Fields are removed by an explicit table, so a configuration field added
later enters every key by default: the worst case is recomputation, never a
wrong reuse. The Accuracy Gate and visualization are never reused across
Experiments, and datasets are shared through their DatasetKey instead.
"""

from collections.abc import Mapping, Sequence
import copy

from ai_qec.utils.hashing import sha256_json, to_jsonable

from .artifact import ArtifactRef

STAGE_INDEPENDENT_FIELDS: Mapping[str, tuple[str, ...]] = {
    "training": (
        "experiment.name",
        "model.decoding",
        "scientific_evaluation",
        "accuracy_gate",
        "performance",
    ),
    "scientific_evaluation": ("experiment.name", "accuracy_gate", "performance"),
    "performance": ("experiment.name", "accuracy_gate"),
}
CROSS_EXPERIMENT_STAGES = tuple(STAGE_INDEPENDENT_FIELDS)
PROVENANCE_KEYS = ("reused_from", "reused_from_experiment", "reuse_key")


def _without(config: Mapping[str, object], paths: Sequence[str]) -> dict:
    pruned = copy.deepcopy(to_jsonable(config))
    for path in paths:
        *parents, leaf = path.split(".")
        node = pruned
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(leaf, None)
    return pruned


def stage_reuse_key(
    stage: str, config: Mapping[str, object], inputs: Sequence[ArtifactRef]
) -> str:
    if stage not in STAGE_INDEPENDENT_FIELDS:
        raise KeyError(f"Stage {stage!r} is never reused across Experiments")
    return sha256_json(
        {
            "schema_version": "stage-reuse-key-v1",
            "stage": stage,
            "inputs": [to_jsonable(ref) for ref in inputs],
            "config": _without(config, STAGE_INDEPENDENT_FIELDS[stage]),
        }
    )
