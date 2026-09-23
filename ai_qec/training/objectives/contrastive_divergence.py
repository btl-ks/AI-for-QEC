"""Contrastive divergence (CD-k) for energy-based models with a free energy."""

from collections.abc import Mapping

import torch

from ai_qec.registry.catalog import LOSSES
from ai_qec.registry.validation import ConfigurationError


class ContrastiveDivergence:
    """``mean F(v_data) - mean F(v_k)``; its gradient is the CD-k estimate.

    The negative sample ``v_k`` comes from ``k`` block-Gibbs steps started at the
    data and is treated as a constant, as in standard CD-k.
    """

    def __init__(self, cd_steps: int) -> None:
        self.cd_steps = cd_steps

    def __call__(self, model, visible, generator):
        with torch.no_grad():
            negative = visible
            for _ in range(self.cd_steps):
                negative = model.sample_visible(model.sample_hidden(negative, generator), generator)
        return model.free_energy(visible).mean() - model.free_energy(negative).mean()


@LOSSES.register("contrastive-divergence")
def build_contrastive_divergence(*, parameters: Mapping[str, object]) -> ContrastiveDivergence:
    unknown = sorted(set(parameters) - {"cd_steps"})
    if unknown:
        raise ConfigurationError(
            f"[training.loss_parameters] unknown keys for contrastive-divergence: {unknown}"
        )
    steps = parameters.get("cd_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ConfigurationError(
            f"[training.loss_parameters.cd_steps] must be a positive integer, got {steps!r}"
        )
    return ContrastiveDivergence(steps)
