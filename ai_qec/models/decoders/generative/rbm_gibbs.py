"""Syndrome-clamped block Gibbs decoding with a joint RBM (Torlai & Melko, Alg. 1).

For every test syndrome ``S0`` an independent chain starts from a random error
layer with the syndrome layer clamped to ``S0``. Each step samples
``h ~ p(h | e, S0)`` and then ``e ~ p(e | h)``. After ``burn_in`` steps the first
``e`` with ``S(e) = S0`` becomes the recovery ``r``. Chains that exhaust
``max_steps`` are reported as timed out; their prediction rows hold ``-1``.
All chains of a request run in parallel on the execution device.
"""

import numpy as np
import torch

from ai_qec.models.decoders.protocol import (
    DecodeRequest,
    DecodeResult,
    DecodeStatus,
    DecoderRuntimeDescriptor,
)
from ai_qec.technology import TechnologyId
from ai_qec.utils.hashing import to_jsonable

INVALID_PREDICTION = -1


class RBMGibbsDecoder:
    def __init__(
        self,
        *,
        decoder_id: str,
        model,
        code,
        burn_in: int,
        max_steps: int,
        pipeline,
        device: str,
        seed: int,
        chunk_size: int = 16_384,
        completion_check_interval: int = 100,
    ) -> None:
        self.decoder_id = decoder_id
        self.model = model
        self.code = code
        self.burn_in = burn_in
        self.max_steps = max_steps
        self.pipeline = pipeline
        self.device = torch.device(device)
        self.seed = seed
        self.chunk_size = chunk_size
        self.completion_check_interval = completion_check_interval

    def runtime(self) -> DecoderRuntimeDescriptor:
        return DecoderRuntimeDescriptor(
            technology_id=TechnologyId.PYTORCH_GPU_DECODER.value,
            technology_version=torch.__version__,
            device=str(self.device),
        )

    def decode(self, request: DecodeRequest) -> DecodeResult:
        batch = request.batch
        syndromes_host = np.asarray(batch.detector_events)
        expected = (self.code.num_checks, self.model.num_syndrome_units)
        if (
            syndromes_host.ndim != 2
            or syndromes_host.shape[1] != expected[0]
            or expected[0] != expected[1]
        ):
            return DecodeResult(
                request_id=request.request_id,
                decoder_id=self.decoder_id,
                status=DecodeStatus.UNSUPPORTED,
                predictions=None,
                runtime=self.runtime(),
                error=f"detector_events shape {syndromes_host.shape} does not match {expected[0]} code checks",
            )
        device_batch, evidence = self.pipeline.transfer(batch)
        syndromes = device_batch.detector_events.float()
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        recoveries, accepted_step = [], []
        with torch.no_grad():
            for start in range(0, syndromes.shape[0], self.chunk_size):
                recovery, step = self._decode_chunk(
                    syndromes[start : start + self.chunk_size], generator
                )
                recoveries.append(recovery.to(torch.uint8).cpu().numpy())
                accepted_step.append(step.cpu().numpy())
        recovery = np.concatenate(recoveries)
        steps = np.concatenate(accepted_step)
        accepted = steps > 0
        if np.any(self.code.syndrome(recovery[accepted]) != syndromes_host[accepted]):
            raise RuntimeError("accepted RBM recovery does not reproduce its syndrome")
        predictions = np.full(
            (len(steps), self.code.num_logicals), INVALID_PREDICTION, dtype=np.int8
        )
        predictions[accepted] = self.code.logical_flips(recovery[accepted]).astype(np.int8)
        timed_out = np.flatnonzero(~accepted)
        accepted_steps = steps[accepted]
        return DecodeResult(
            request_id=request.request_id,
            decoder_id=self.decoder_id,
            status=DecodeStatus.TIMED_OUT if timed_out.size else DecodeStatus.SUCCEEDED,
            predictions=predictions,
            runtime=self.runtime(),
            failed_sample_ids=tuple(batch.sample_ids[int(index)] for index in timed_out),
            error=None
            if not timed_out.size
            else f"{timed_out.size} chains found no compatible error chain within {self.max_steps} steps",
            provenance={
                "algorithm": "torlai-melko-2017-algorithm-1",
                "burn_in": self.burn_in,
                "max_steps": self.max_steps,
                "seed": self.seed,
                "accepted": int(accepted.sum()),
                "timed_out": int(timed_out.size),
                "acceptance_steps": {
                    "mean": float(accepted_steps.mean()) if accepted_steps.size else None,
                    "median": float(np.median(accepted_steps)) if accepted_steps.size else None,
                    "p95": float(np.percentile(accepted_steps, 95))
                    if accepted_steps.size
                    else None,
                    "max": int(accepted_steps.max()) if accepted_steps.size else None,
                },
                "transfer": to_jsonable(evidence),
            },
        )

    def _decode_chunk(self, syndromes, generator):
        model = self.model
        weight = model.error_weight
        error_bias = model.error_bias
        parity = torch.tensor(
            np.array(self.code.parity_check.T), dtype=torch.float32, device=self.device
        )
        clamped_field = syndromes @ model.syndrome_weight.T + model.hidden_bias
        rows = syndromes.shape[0]
        errors = torch.bernoulli(
            torch.full((rows, model.num_error_units), 0.5, device=self.device), generator=generator
        )
        recovery = torch.zeros_like(errors)
        step_found = torch.zeros(rows, dtype=torch.int64, device=self.device)
        found = torch.zeros(rows, dtype=torch.bool, device=self.device)
        for step in range(1, self.max_steps + 1):
            hidden = torch.bernoulli(
                torch.sigmoid(errors @ weight.T + clamped_field), generator=generator
            )
            errors = torch.bernoulli(
                torch.sigmoid(hidden @ weight + error_bias), generator=generator
            )
            if step <= self.burn_in:
                continue
            compatible = torch.all(torch.remainder(errors @ parity, 2) == syndromes, dim=1) & ~found
            recovery = torch.where(compatible[:, None], errors, recovery)
            step_found = torch.where(compatible, step, step_found)
            found |= compatible
            if step % self.completion_check_interval == 0 and bool(found.all()):
                break
        return recovery, step_found
