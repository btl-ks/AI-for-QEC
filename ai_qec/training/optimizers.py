"""Optimizers and learning-rate schedules selected by configuration."""

from collections.abc import Iterable, Mapping

import torch

from ai_qec.registry.catalog import OPTIMIZERS, SCHEDULERS
from ai_qec.registry.validation import ConfigurationError


@OPTIMIZERS.register("sgd")
def build_sgd(
    *,
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float,
    options: Mapping[str, object],
) -> torch.optim.SGD:
    unknown = sorted(set(options) - {"weight_decay", "momentum"})
    if unknown:
        raise ConfigurationError(f"[training.optimizer_parameters] unknown keys for sgd: {unknown}")
    values = {name: options.get(name, 0.0) for name in ("weight_decay", "momentum")}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigurationError(
                f"[training.optimizer_parameters.{name}] must be >= 0, got {value!r}"
            )
    if not learning_rate > 0:
        raise ConfigurationError(
            f"[training.learning_rate] must be positive, got {learning_rate!r}"
        )
    return torch.optim.SGD(
        parameters,
        lr=learning_rate,
        weight_decay=float(values["weight_decay"]),
        momentum=float(values["momentum"]),
    )


def _constant_factor(epoch: int) -> float:
    return 1.0


@SCHEDULERS.register("constant")
def build_constant_schedule(
    *, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _constant_factor)
