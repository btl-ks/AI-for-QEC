"""Synthetic simulator used by the initial runnable project slice."""

from __future__ import annotations

from typing import Any

import numpy as np

from ai_qec.qec.circuits.base import QECCircuit
from ai_qec.qec.detectors.syndrome import FEATURE_NAMES, detector_summary_features
from ai_qec.qec.noise.base import NoiseModel
from ai_qec.qec.simulators.base import QECBackend


class ToySyntheticQECBackend(QECBackend):
    """Generate detector events with weak pair-correlation crosstalk."""

    name = "toy_synthetic"

    def __init__(self, circuit: QECCircuit, noise: NoiseModel, config: dict[str, Any]) -> None:
        """Store circuit/noise context and synthetic control settings."""
        self.circuit = circuit
        self.noise = noise
        controls = config.get("noise", {}).get("controls", {})
        self.match_density = bool(controls.get("match_total_syndrome_density", True))
        self.crosstalk_gain = float(config.get("data", {}).get("synthetic_crosstalk_gain", 3.0))

    def sample_batch(self, n_samples: int, rng: np.random.Generator) -> dict[str, Any]:
        """Generate non-empty detector-summary features and labels."""
        draw = self.noise.sample(n_samples, rng)
        rounds = self.circuit.rounds
        detectors = self.circuit.code.num_stabilizers_per_round

        base_p = 4.5 * draw.depolarizing_rate + 3.0 * draw.measurement_rate
        # Follow this branch when self.match_density.
        if self.match_density:
            base_p = base_p - 0.55 * self.crosstalk_gain * draw.target_strength
        base_p = np.clip(base_p, 1e-5, 0.25)

        events = rng.random((n_samples, rounds, detectors)) < base_p[:, None, None]

        # Follow this branch when detectors > 1.
        if detectors > 1:
            pair_p = np.clip(self.crosstalk_gain * draw.target_strength, 0.0, 0.12)
            pair_mask = rng.random((n_samples, rounds, detectors - 1)) < pair_p[:, None, None]
            events[:, :, 1:] |= pair_mask
            events[:, :, :-1] |= pair_mask

        # Follow this branch when rounds > 1.
        if rounds > 1:
            temporal_p = np.clip(0.35 * self.crosstalk_gain * draw.target_strength, 0.0, 0.05)
            temporal_mask = rng.random((n_samples, rounds - 1, detectors)) < temporal_p[:, None, None]
            events[:, 1:, :] |= temporal_mask & events[:, :-1, :]

        features = detector_summary_features(events)
        density = features[:, FEATURE_NAMES.index("density")]
        adjacent = features[:, FEATURE_NAMES.index("adjacent_pair_rate")]

        logical_logit = -3.3 + 42.0 * density + 26.0 * adjacent + 90.0 * draw.target_strength
        logical_probability = 1.0 / (1.0 + np.exp(-logical_logit))
        logical_label = (rng.random(n_samples) < logical_probability).astype(np.int64)

        return {
            "features": features.astype(np.float64),
            "target_strength": draw.target_strength,
            "logical_label": logical_label,
            "depolarizing_rate": draw.depolarizing_rate,
            "measurement_rate": draw.measurement_rate,
            "logical_probability": logical_probability.astype(np.float64),
        }


# Compatibility import only. New code must use the fidelity-explicit name.
SyntheticQECBackend = ToySyntheticQECBackend
