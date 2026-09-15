"""Latency benchmark helper."""

from __future__ import annotations

import numpy as np


def latency_summary(seconds: list[float]) -> dict[str, float]:
    """Summarize latency samples with mean and tail percentiles."""
    values = np.asarray(seconds, dtype=np.float64)
    return {
        "latency/mean": float(np.mean(values)),
        "latency/p50": float(np.percentile(values, 50)),
        "latency/p95": float(np.percentile(values, 95)),
        "latency/p99": float(np.percentile(values, 99)),
    }
