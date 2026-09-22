"""Batch decoding throughput on a fixed shot set.

This is a software measurement of whole-batch decode time with explicit
synchronization boundaries. It is not a real-time latency study.
"""

from collections.abc import Callable, Mapping
import statistics
import time

from ai_qec.models.decoders.protocol import DecodeRequest
from ai_qec.utils.hashing import to_jsonable

REPORT_SCHEMA = "batch-throughput-report-v1"


def measure_batch_throughput(
    decoders: Mapping[str, object],
    batch,
    *,
    repetitions: int,
    warmup: int,
    synchronize: Callable[[], None],
    environment: Mapping[str, object],
) -> dict[str, object]:
    shots = len(batch.sample_ids)
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "measurement": "whole-batch decode wall time between device synchronizations",
        "shots": shots,
        "repetitions": repetitions,
        "warmup": warmup,
        "environment": dict(environment),
        "decoders": {},
    }
    for decoder_id, decoder in decoders.items():
        request = DecodeRequest(
            request_id=f"performance.{decoder_id}", decoder_id=decoder_id, batch=batch
        )
        for _ in range(warmup):
            decoder.decode(request)
        seconds = []
        runtime = None
        for _ in range(repetitions):
            synchronize()
            started = time.perf_counter()
            result = decoder.decode(request)
            synchronize()
            seconds.append(time.perf_counter() - started)
            runtime = result.runtime
        mean = statistics.fmean(seconds)
        report["decoders"][decoder_id] = {
            "runtime": to_jsonable(runtime),
            "seconds": seconds,
            "mean_seconds": mean,
            "min_seconds": min(seconds),
            "shots_per_second": shots / mean,
            "mean_seconds_per_shot": mean / shots,
        }
    return report
