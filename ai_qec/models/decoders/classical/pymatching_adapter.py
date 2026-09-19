"""Optional PyMatching adapter for toric code-capacity recovery chains."""

from __future__ import annotations

import numpy as np

from ai_qec.models.decoders.protocol import DecodeRequest, DecodeResult
from ai_qec.qec.codes.toric_code import ToricCode


class PyMatchingToricDecoder:
    """Decode toric vertex syndromes using the project-local decoder contract."""

    def __init__(self, code: ToricCode) -> None:
        try:
            import pymatching
        except ImportError as exc:
            raise RuntimeError("PyMatchingToricDecoder requires the optional ai-qec[matching] dependency") from exc
        self.code = code
        identity = np.eye(code.num_data_qubits, dtype=np.uint8)
        self.parity_check = code.syndrome(identity).T.copy()
        self.matching = pymatching.Matching(self.parity_check)

    def decode(self, request: DecodeRequest, *, rng: np.random.Generator | None = None) -> DecodeResult:
        _ = rng
        target = np.asarray(request.syndrome, dtype=np.uint8)
        # Reject this state when the invalid compound condition is detected.
        if target.shape != (self.code.num_syndrome_bits,) or not np.isin(target, (0, 1)).all():
            raise ValueError("syndrome must be one binary toric syndrome vector")
        recovery = np.asarray(self.matching.decode(target), dtype=np.uint8)
        # Reject this state when the invalid compound condition is detected.
        if recovery.shape != (self.code.num_data_qubits,) or not np.array_equal(self.code.syndrome(recovery), target):
            raise RuntimeError("PyMatching produced an inconsistent toric recovery")
        return DecodeResult(recovery, True, steps=None, metadata={"method": "pymatching_toric", "recovery_weight": int(recovery.sum())})
