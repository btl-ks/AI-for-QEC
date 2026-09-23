"""PyMatching minimum-weight perfect matching baseline (``pymatching-cpu-decoder``)."""

import numpy as np
import pymatching

from ai_qec.models.decoders.protocol import (
    DecodeRequest,
    DecodeResult,
    DecodeStatus,
    DecoderRuntimeDescriptor,
)
from ai_qec.registry.catalog import DECODERS
from ai_qec.technology import TechnologyId


class PyMatchingDecoder:
    """Uniform edge weights, i.e. the Manhattan distance used by the paper's MWPM."""

    decoder_id = TechnologyId.PYMATCHING_CPU_DECODER.value

    def __init__(self, *, code) -> None:
        self.code = code
        self._matching = pymatching.Matching.from_check_matrix(
            np.asarray(code.parity_check), faults_matrix=np.asarray(code.logical_operators)
        )

    def runtime(self) -> DecoderRuntimeDescriptor:
        return DecoderRuntimeDescriptor(
            technology_id=self.decoder_id,
            technology_version=pymatching.__version__,
            device="cpu",
        )

    def decode(self, request: DecodeRequest) -> DecodeResult:
        syndromes = np.asarray(request.batch.detector_events, dtype=np.uint8)
        if syndromes.ndim != 2 or syndromes.shape[1] != self.code.num_checks:
            return DecodeResult(
                request_id=request.request_id,
                decoder_id=self.decoder_id,
                status=DecodeStatus.UNSUPPORTED,
                predictions=None,
                runtime=self.runtime(),
                error=f"detector_events shape {syndromes.shape} does not match {self.code.num_checks} code checks",
            )
        predictions = self._matching.decode_batch(syndromes).astype(np.int8)
        return DecodeResult(
            request_id=request.request_id,
            decoder_id=self.decoder_id,
            status=DecodeStatus.SUCCEEDED,
            predictions=predictions,
            runtime=self.runtime(),
            provenance={"edge_weights": "uniform", "matching": "minimum-weight perfect matching"},
        )


@DECODERS.register(TechnologyId.PYMATCHING_CPU_DECODER.value)
def build_pymatching_decoder(*, code) -> PyMatchingDecoder:
    return PyMatchingDecoder(code=code)
