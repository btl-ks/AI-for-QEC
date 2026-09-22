"""Explicitly import the executable implementations so they register themselves.

Importing :mod:`ai_qec.notebook_api` never loads these modules; only a runtime
that is about to execute (for example ``LocalNotebookPlatform``) calls
:func:`load_builtin_implementations`. Each module registers its factories with
the Registry decorators exactly once, because Python caches imported modules.
"""

import importlib

from ai_qec.registries import REGISTRIES_BY_PATH

IMPLEMENTATION_MODULES = (
    "ai_qec.qec.codes.toric",
    "ai_qec.qec.noise_models",
    "ai_qec.qec.backends.stim_noise",
    "ai_qec.data.generators.stim_code_capacity",
    "ai_qec.data.loaders.pytorch",
    "ai_qec.models.generative.rbm",
    "ai_qec.models.decoders.classical.pymatching_adapter",
    "ai_qec.training.objectives.contrastive_divergence",
    "ai_qec.training.optimizers",
    "ai_qec.training.trainers.pytorch_trainer",
)


def load_builtin_implementations() -> dict[str, tuple[str, ...]]:
    """Import every implementation module and return the registered keys per path."""

    for module in IMPLEMENTATION_MODULES:
        importlib.import_module(module)
    return {path: registry.keys() for path, registry in REGISTRIES_BY_PATH.items()}
