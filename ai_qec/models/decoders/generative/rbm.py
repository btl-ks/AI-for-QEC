"""PyTorch restricted Boltzmann machine for joint error--syndrome samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class JointErrorSyndromeRBM(nn.Module):
    """Binary RBM with visible units ordered as ``[error | syndrome]``."""

    def __init__(self, *, error_units: int, syndrome_units: int, hidden_units: int, init_width: float, seed: int) -> None:
        super().__init__()
        if min(error_units, syndrome_units, hidden_units) < 1 or init_width <= 0:
            raise ValueError("RBM widths and init_width must be positive")
        self.error_units = int(error_units)
        self.syndrome_units = int(syndrome_units)
        self.hidden_units = int(hidden_units)
        self.visible_units = self.error_units + self.syndrome_units
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.weights = nn.Parameter(torch.randn(self.visible_units, self.hidden_units, generator=generator) * init_width)
        self.visible_bias = nn.Parameter(torch.zeros(self.visible_units))
        self.hidden_bias = nn.Parameter(torch.zeros(self.hidden_units))

    def _visible(self, visible: np.ndarray | torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(visible, dtype=self.weights.dtype, device=self.weights.device)
        if values.ndim != 2 or values.shape[1] != self.visible_units:
            raise ValueError(f"visible must have shape (n, {self.visible_units})")
        if not torch.all((values == 0) | (values == 1)):
            raise ValueError("RBM visible states must be binary")
        return values

    def hidden_probabilities(self, visible: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self._visible(visible) @ self.weights + self.hidden_bias)

    def visible_probabilities(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_units:
            raise ValueError("hidden has an incompatible shape")
        return torch.sigmoid(hidden @ self.weights.T + self.visible_bias)

    def error_probabilities(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.visible_probabilities(hidden)[:, :self.error_units]

    def random_error_chains(self, n_chains: int, generator: torch.Generator) -> torch.Tensor:
        """Draw uniformly random binary error chains as Gibbs starting states."""
        return torch.randint(0, 2, (n_chains, self.error_units), generator=generator, device=self.weights.device).to(self.weights.dtype)

    def sample_hidden(self, error: torch.Tensor, syndrome: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Sample ``h ~ p(h | e, S)`` with the same clamped syndrome for every chain."""
        clamped = syndrome.to(self.weights.dtype).expand(len(error), -1)
        return self._draw(self.hidden_probabilities(torch.cat((error, clamped), dim=1)), generator)

    def sample_error(self, hidden: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Sample ``e ~ p(e | h)``; the syndrome units stay clamped and are not resampled."""
        return self._draw(self.error_probabilities(hidden), generator)

    @staticmethod
    def _draw(probability: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        return (torch.rand(probability.shape, generator=generator, device=probability.device) < probability).to(probability.dtype)

    def to_visible_tensor(self, visible: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Validate a whole split once and place it on the model device for ``validated=True`` steps."""
        return self._visible(visible)

    def contrastive_divergence_step(
        self, visible: np.ndarray | torch.Tensor, *, optimizer: torch.optim.Optimizer,
        cd_steps: int, generator: torch.Generator, validated: bool = False, sync: bool = True,
    ) -> float | torch.Tensor:
        """Use CD-k to estimate the RBM likelihood gradient and update parameters.

        ``validated=True`` accepts a slice of :meth:`to_visible_tensor` and skips the per-batch
        binary check; ``sync=False`` returns the reconstruction BCE as a detached device tensor
        instead of a float.  Neither option changes the update or the random stream.
        """
        if cd_steps < 1:
            raise ValueError("cd_steps must be positive")
        if validated:
            if not (isinstance(visible, torch.Tensor) and visible.device == self.weights.device
                    and visible.dtype == self.weights.dtype and visible.ndim == 2
                    and visible.shape[1] == self.visible_units):
                raise ValueError("validated=True requires a slice of to_visible_tensor()")
            positive = visible
        else:
            positive = self._visible(visible)
        if len(positive) == 0:
            raise ValueError("cannot train on an empty minibatch")
        with torch.no_grad():
            positive_hidden = self.hidden_probabilities(positive)
            hidden = self._draw(positive_hidden, generator)
            negative = positive
            for _ in range(cd_steps):
                negative = self._draw(self.visible_probabilities(hidden), generator)
                hidden = self._draw(self.hidden_probabilities(negative), generator)
        optimizer.zero_grad(set_to_none=True)
        positive_energy = -(torch.sum((positive @ self.weights) * positive_hidden, dim=1) + positive @ self.visible_bias + positive_hidden @ self.hidden_bias).mean()
        negative_hidden = self.hidden_probabilities(negative).detach()
        negative_energy = -(torch.sum((negative @ self.weights) * negative_hidden, dim=1) + negative @ self.visible_bias + negative_hidden @ self.hidden_bias).mean()
        (positive_energy - negative_energy).backward()
        optimizer.step()
        loss = self._reconstruction_bce_tensor(positive)
        return float(loss.item()) if sync else loss

    @torch.no_grad()
    def _reconstruction_bce_tensor(self, values: torch.Tensor) -> torch.Tensor:
        reconstructed = torch.sigmoid(torch.sigmoid(values @ self.weights + self.hidden_bias) @ self.weights.T + self.visible_bias)
        return F.binary_cross_entropy(reconstructed, values)

    @torch.no_grad()
    def reconstruction_bce(self, visible: np.ndarray | torch.Tensor) -> float:
        return float(self._reconstruction_bce_tensor(self._visible(visible)).item())

    def save(self, path: str | Path, *, metadata: dict[str, Any], optimizer: torch.optim.Optimizer | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": 1,
            "dimensions": (self.error_units, self.syndrome_units, self.hidden_units),
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "metadata": metadata,
        }, target)

    @classmethod
    def load_with_metadata(cls, path: str | Path) -> tuple["JointErrorSyndromeRBM", dict[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if set(payload) != {"schema_version", "dimensions", "model_state", "optimizer_state", "metadata"} or payload["schema_version"] != 1:
            raise ValueError("RBM checkpoint schema mismatch")
        dimensions = payload["dimensions"]
        if not isinstance(dimensions, (tuple, list)) or len(dimensions) != 3 or not all(isinstance(x, int) and x > 0 for x in dimensions):
            raise ValueError("RBM checkpoint dimensions are invalid")
        model = cls(error_units=dimensions[0], syndrome_units=dimensions[1], hidden_units=dimensions[2], init_width=0.01, seed=0)
        model.load_state_dict(payload["model_state"], strict=True)
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("RBM checkpoint metadata must be an object")
        model.eval()
        return model, metadata
