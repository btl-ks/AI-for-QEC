"""Syndrome-clamped PyTorch Gibbs decoding for the joint RBM."""

from __future__ import annotations

import numpy as np
import torch

from ai_qec.models.decoders.generative.rbm import JointErrorSyndromeRBM
from ai_qec.models.decoders.protocol import DecodeRequest, DecodeResult
from ai_qec.qec.codes.toric_code import ToricCode


GibbsDecodeResult = DecodeResult


class RBMGibbsDecoder:
    """Sample one or more error chains while holding the measured syndrome fixed."""

    def __init__(
        self, model: JointErrorSyndromeRBM, code: ToricCode, *, burn_in: int,
        max_steps: int, parallel_chains: int = 1, device: str = "cpu",
    ) -> None:
        if model.error_units != code.num_data_qubits or model.syndrome_units != code.num_syndrome_bits:
            raise ValueError("RBM dimensions do not match the toric code")
        if burn_in < 0 or max_steps < 1 or burn_in >= max_steps or parallel_chains < 1:
            raise ValueError("require 0 <= burn_in < max_steps and parallel_chains >= 1")
        if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError(f"PyTorch device is unavailable: {device}")
        self.model = model.to(device).eval()
        self.code = code
        self.burn_in = int(burn_in)
        self.max_steps = int(max_steps)
        self.parallel_chains = int(parallel_chains)
        self.device = device
        # H maps edge-error vectors to vertex syndromes over GF(2).
        identity = np.eye(code.num_data_qubits, dtype=np.uint8)
        self.parity_check = torch.as_tensor(code.syndrome(identity).T.copy(), dtype=torch.int64, device=device)

    @torch.no_grad()
    def decode(self, syndrome: np.ndarray | DecodeRequest, *, rng: np.random.Generator | None = None) -> DecodeResult:
        """Return the first compatible recovery after burn-in, or a timeout."""
        request = syndrome if isinstance(syndrome, DecodeRequest) else DecodeRequest(syndrome)
        target_np = np.asarray(request.syndrome, dtype=np.uint8)
        if target_np.shape != (self.code.num_syndrome_bits,) or not np.isin(target_np, (0, 1)).all():
            raise ValueError("syndrome must be one binary toric syndrome vector")
        rng = rng if rng is not None else np.random.default_rng()
        generator = torch.Generator(device=self.device).manual_seed(int(rng.integers(0, 2**63 - 1)))
        target = torch.as_tensor(target_np, dtype=torch.int64, device=self.device)
        error = torch.randint(0, 2, (self.parallel_chains, self.code.num_data_qubits), generator=generator, device=self.device).to(self.model.weights.dtype)
        clamped = target.to(self.model.weights.dtype).expand(self.parallel_chains, -1)
        for step in range(1, self.max_steps + 1):
            visible = torch.cat((error, clamped), dim=1)
            hidden = self.model._draw(self.model.hidden_probabilities(visible), generator)
            error = self.model._draw(self.model.error_probabilities(hidden), generator)
            if step <= self.burn_in:
                continue
            sampled_syndrome = torch.remainder(error.to(torch.int64) @ self.parity_check.T, 2)
            compatible = torch.all(sampled_syndrome == target, dim=1)
            matches = torch.nonzero(compatible, as_tuple=False)
            if matches.numel():
                chain = int(matches[0, 0].item())
                recovery = error[chain].to(torch.uint8).cpu().numpy()
                return DecodeResult(recovery, True, steps=step, metadata={"parallel_chains": self.parallel_chains, "accepted_chain": chain, "device": self.device})
        return DecodeResult(None, False, steps=self.max_steps, metadata={"parallel_chains": self.parallel_chains, "device": self.device})
