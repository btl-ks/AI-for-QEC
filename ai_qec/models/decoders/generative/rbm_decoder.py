"""Syndrome-clamped PyTorch Gibbs decoding for the joint RBM."""

from __future__ import annotations

import numpy as np
import torch

from ai_qec.models.decoders.generative.rbm import JointErrorSyndromeRBM
from ai_qec.models.decoders.protocol import DecodeRequest, DecodeResult
from ai_qec.qec.codes.toric_code import ToricCode


GibbsDecodeResult = DecodeResult


def torch_generator_from(rng: np.random.Generator, device: str) -> torch.Generator:
    """Derive a seeded PyTorch generator from a NumPy generator."""
    return torch.Generator(device=device).manual_seed(int(rng.integers(0, 2**63 - 1)))


def first_compatible_chain(error: torch.Tensor, target: torch.Tensor, parity_check: torch.Tensor) -> int | None:
    """Return the first chain whose syndrome ``H e mod 2`` equals ``target``, or None."""
    # Each toric vertex check touches four edges, so these binary sums are
    # exact in float32. CUDA does not implement integer matrix multiplication.
    syndrome = torch.remainder(error.to(torch.float32) @ parity_check.T.to(torch.float32), 2)
    compatible = torch.all(syndrome == target.to(torch.float32), dim=1)
    matches = torch.nonzero(compatible, as_tuple=False)
    return int(matches[0, 0].item()) if matches.numel() else None


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
        self.parity_check = torch.as_tensor(code.parity_check_matrix(), dtype=torch.int64, device=device)

    @torch.no_grad()
    def decode(self, syndrome: np.ndarray | DecodeRequest, *, rng: np.random.Generator | None = None) -> DecodeResult:
        """Return the first compatible recovery after burn-in, or a timeout."""
        request = syndrome if isinstance(syndrome, DecodeRequest) else DecodeRequest(syndrome)
        target_np = np.asarray(request.syndrome, dtype=np.uint8)
        if target_np.shape != (self.code.num_syndrome_bits,) or not np.isin(target_np, (0, 1)).all():
            raise ValueError("syndrome must be one binary toric syndrome vector")
        rng = rng if rng is not None else np.random.default_rng()
        generator = torch_generator_from(rng, self.device)
        target = torch.as_tensor(target_np, dtype=torch.int64, device=self.device)
        error = self.model.random_error_chains(self.parallel_chains, generator)
        for step in range(1, self.max_steps + 1):
            hidden = self.model.sample_hidden(error, target, generator)
            error = self.model.sample_error(hidden, generator)
            if step <= self.burn_in:
                continue
            chain = first_compatible_chain(error, target, self.parity_check)
            if chain is not None:
                recovery = error[chain].to(torch.uint8).cpu().numpy()
                return DecodeResult(recovery, True, steps=step, metadata={"parallel_chains": self.parallel_chains, "accepted_chain": chain, "device": self.device})
        return DecodeResult(None, False, steps=self.max_steps, metadata={"parallel_chains": self.parallel_chains, "device": self.device})
