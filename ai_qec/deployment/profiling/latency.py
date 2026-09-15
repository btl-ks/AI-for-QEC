"""Latency profiling."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def time_call(fn: Callable[[], T]) -> tuple[T, float]:
    """Run a zero-argument callable and return its result plus elapsed seconds."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start
