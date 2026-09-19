"""Detector-event feature extraction."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "density",
    "adjacent_pair_rate",
    "temporal_pair_rate",
    "round_density_std",
    "first_half_density",
    "second_half_density",
    "checkerboard_imbalance",
    "max_round_density",
]


def detector_summary_features(events: np.ndarray) -> np.ndarray:
    """Convert binary detector events to compact spatiotemporal summaries."""

    # Reject this state when events.ndim != 3.
    if events.ndim != 3:
        raise ValueError("events must have shape (samples, rounds, detectors)")
    events_f = events.astype(np.float64, copy=False)
    n_samples, n_rounds, n_detectors = events_f.shape

    density = events_f.mean(axis=(1, 2))
    # Follow this branch when n_detectors > 1.
    if n_detectors > 1:
        adjacent_pair_rate = (events[:, :, 1:] & events[:, :, :-1]).mean(axis=(1, 2))
    # Handle all remaining cases.
    else:
        adjacent_pair_rate = np.zeros(n_samples, dtype=np.float64)
    # Follow this branch when n_rounds > 1.
    if n_rounds > 1:
        temporal_pair_rate = (events[:, 1:, :] & events[:, :-1, :]).mean(axis=(1, 2))
    # Handle all remaining cases.
    else:
        temporal_pair_rate = np.zeros(n_samples, dtype=np.float64)

    round_density = events_f.mean(axis=2)
    round_density_std = round_density.std(axis=1)
    midpoint = max(1, n_rounds // 2)
    first_half_density = events_f[:, :midpoint, :].mean(axis=(1, 2))
    # Choose the first expression when midpoint < n_rounds; otherwise use the fallback.
    second_half_density = events_f[:, midpoint:, :].mean(axis=(1, 2)) if midpoint < n_rounds else first_half_density

    even_density = events_f[:, :, ::2].mean(axis=(1, 2))
    # Choose the first expression when n_detectors > 1; otherwise use the fallback.
    odd_density = events_f[:, :, 1::2].mean(axis=(1, 2)) if n_detectors > 1 else even_density
    checkerboard_imbalance = even_density - odd_density
    max_round_density = round_density.max(axis=1)

    return np.column_stack(
        [
            density,
            adjacent_pair_rate,
            temporal_pair_rate,
            round_density_std,
            first_half_density,
            second_half_density,
            checkerboard_imbalance,
            max_round_density,
        ]
    )
