"""Exact minimum-weight perfect matching reference for small toric lattices."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ai_qec.qec.codes.toric_code import ToricCode
from ai_qec.models.decoders.protocol import DecodeRequest, DecodeResult


class ExactToricMWPMDecoder:
    """Minimum-weight perfect matching over toric syndrome defects.

    This dependency-free reference enumerates the exact matching by dynamic
    programming.  It is appropriate for the small smoke lattices and guards a
    configured defect limit instead of silently substituting a greedy matching
    when a larger production lattice needs a Blossom implementation.
    """

    def __init__(self, code: ToricCode, *, max_exact_defects: int = 18) -> None:
        if max_exact_defects < 2 or max_exact_defects % 2:
            raise ValueError("max_exact_defects must be an even integer >= 2")
        self.code = code
        self.max_exact_defects = max_exact_defects

    def decode(self, syndrome: np.ndarray | DecodeRequest, *, rng: np.random.Generator | None = None) -> np.ndarray | DecodeResult:
        """Return an exact minimum-weight recovery for a valid toric syndrome."""
        request = syndrome if isinstance(syndrome, DecodeRequest) else None
        if request is not None:
            syndrome = request.syndrome
        _ = rng
        target = np.asarray(syndrome, dtype=np.uint8)
        if target.shape != (self.code.num_syndrome_bits,) or not np.isin(target, (0, 1)).all():
            raise ValueError("syndrome must be a binary toric syndrome vector")
        defects = [(int(x), int(y)) for y, x in np.argwhere(target.reshape(self.code.distance, self.code.distance))]
        if len(defects) % 2:
            raise ValueError("toric syndrome must contain an even number of defects")
        if len(defects) > self.max_exact_defects:
            raise RuntimeError(
                f"exact MWPM is limited to {self.max_exact_defects} defects; install a scalable matching backend for this syndrome"
            )
        if not defects:
            recovery = np.zeros(self.code.num_data_qubits, dtype=np.uint8)
            return DecodeResult(recovery, True, steps=0, metadata={"method": "exact_toric_mwpm"}) if request is not None else recovery

        pairs = self._minimum_pairs(defects)
        recovery = np.zeros((2, self.code.distance, self.code.distance), dtype=np.uint8)
        for first, second in pairs:
            self._add_shortest_path(recovery, defects[first], defects[second])
        flat = np.concatenate((recovery[0].reshape(-1), recovery[1].reshape(-1)))
        if not np.array_equal(self.code.syndrome(flat), target):
            raise RuntimeError("MWPM path construction produced an inconsistent recovery")
        return DecodeResult(flat, True, steps=None, metadata={"method": "exact_toric_mwpm"}) if request is not None else flat

    def _minimum_pairs(self, defects: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        count = len(defects)

        @lru_cache(maxsize=None)
        def solve(mask: int) -> tuple[int, tuple[tuple[int, int], ...]]:
            if mask == 0:
                return 0, ()
            first = (mask & -mask).bit_length() - 1
            remaining = mask ^ (1 << first)
            best_cost = 10**9
            best_pairs: tuple[tuple[int, int], ...] = ()
            candidate_mask = remaining
            while candidate_mask:
                bit = candidate_mask & -candidate_mask
                second = bit.bit_length() - 1
                tail_cost, tail_pairs = solve(remaining ^ bit)
                cost = self._distance(defects[first], defects[second]) + tail_cost
                pair_tuple = ((first, second),) + tail_pairs
                if cost < best_cost or (cost == best_cost and pair_tuple < best_pairs):
                    best_cost, best_pairs = cost, pair_tuple
                candidate_mask ^= bit
            return best_cost, best_pairs

        return solve((1 << count) - 1)[1]

    def _distance(self, first: tuple[int, int], second: tuple[int, int]) -> int:
        return self._periodic_delta(first[0], second[0]) + self._periodic_delta(first[1], second[1])

    def _periodic_delta(self, start: int, stop: int) -> int:
        forward = (stop - start) % self.code.distance
        return min(forward, self.code.distance - forward)

    def _add_shortest_path(self, recovery: np.ndarray, first: tuple[int, int], second: tuple[int, int]) -> None:
        x, y = first
        target_x, target_y = second
        while x != target_x:
            forward = (target_x - x) % self.code.distance
            backward = (x - target_x) % self.code.distance
            if forward <= backward:
                recovery[0, y, x] ^= 1
                x = (x + 1) % self.code.distance
            else:
                recovery[0, y, (x - 1) % self.code.distance] ^= 1
                x = (x - 1) % self.code.distance
        while y != target_y:
            forward = (target_y - y) % self.code.distance
            backward = (y - target_y) % self.code.distance
            if forward <= backward:
                recovery[1, y, x] ^= 1
                y = (y + 1) % self.code.distance
            else:
                recovery[1, (y - 1) % self.code.distance, x] ^= 1
                y = (y - 1) % self.code.distance
