"""Reproducible CUDA Graph training-step benchmark retained as change evidence.

This benchmark deliberately keeps the production DataLoader contract unchanged.
The microbenchmark uses a bounded set of distinct device minibatches to isolate
step submission, while the Stage benchmark uses LocalNotebookPlatform and its
normal per-minibatch host-to-device path.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path


def _context():
    import torch

    from ai_qec.models.generative.rbm import JointErrorSyndromeRBMFamily
    from ai_qec.models.spec import ModelSpec
    from ai_qec.training.executors import TrainingStepContext
    from ai_qec.training.objectives.contrastive_divergence import ContrastiveDivergence

    family = JointErrorSyndromeRBMFamily()
    model = family.create(
        ModelSpec(
            "joint-error-syndrome-rbm",
            "1",
            {"hidden_units": 128, "init_std": 0.01},
        ),
        widths={"physical_errors": 72, "detector_events": 36},
        seed=17,
    ).to("cuda:0")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    generator = torch.Generator(device="cuda:0").manual_seed(29)
    return TrainingStepContext(
        model=model,
        visible=family.visible,
        objective=ContrastiveDivergence(10),
        optimizer=optimizer,
        generator=generator,
        device="cuda:0",
    )


def _microbenchmark(steps: int = 1_000) -> dict[str, object]:
    import torch

    from ai_qec.training.executors.cuda_graph import (
        build_pytorch_cuda_graph_training_step_executor,
    )
    from ai_qec.training.executors.eager import build_pytorch_eager_training_step_executor

    batches = [
        {
            "physical_errors": torch.randint(0, 2, (100, 72), device="cuda:0", dtype=torch.uint8),
            "detector_events": torch.randint(0, 2, (100, 36), device="cuda:0", dtype=torch.uint8),
        }
        for _ in range(64)
    ]

    eager = build_pytorch_eager_training_step_executor(
        device="cuda:0", options={}, context=_context()
    )
    for index in range(20):
        eager.step(batches[index % len(batches)])
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index in range(steps):
        eager.step(batches[index % len(batches)])
    torch.cuda.synchronize()
    eager_seconds = time.perf_counter() - started

    graph = build_pytorch_cuda_graph_training_step_executor(
        device="cuda:0",
        options={"max_graphs": 1},
        context=_context(),
    )
    graph.step(batches[0])
    graph.step(batches[1])
    for index in range(20):
        graph.step(batches[(index + 2) % len(batches)])
    torch.cuda.synchronize()
    before = graph.evidence()
    started = time.perf_counter()
    for index in range(steps):
        graph.step(batches[index % len(batches)])
    torch.cuda.synchronize()
    graph_seconds = time.perf_counter() - started
    graph.finalize()
    after = graph.evidence()

    static = next(iter(graph._states.values())).static_inputs
    copy_start = torch.cuda.Event(enable_timing=True)
    copy_end = torch.cuda.Event(enable_timing=True)
    copy_start.record()
    for index in range(steps):
        batch = batches[index % len(batches)]
        for field, target in static.items():
            target.copy_(batch[field], non_blocking=True)
    copy_end.record()
    copy_end.synchronize()

    return {
        "shape": {"batch": 100, "physical_errors": 72, "detector_events": 36},
        "hidden_units": 128,
        "cd_steps": 10,
        "distinct_device_batches": len(batches),
        "timed_steps": steps,
        "eager_step_ms": eager_seconds * 1_000 / steps,
        "batch_d2d_copy_ms": copy_start.elapsed_time(copy_end) / steps,
        "capture_ms": after.capture_seconds * 1_000,
        "steady_copy_plus_replay_ms": graph_seconds * 1_000 / steps,
        "steady_replay_host_submit_ms": (
            (after.replay_seconds - before.replay_seconds) * 1_000 / steps
        ),
        "speedup": eager_seconds / graph_seconds,
        "evidence": {
            "warmup_steps": after.warmup_steps,
            "capture_count": after.capture_count,
            "replay_steps": after.replay_steps,
            "fallback_observed": after.fallback_observed,
        },
    }


def _config(executor: str) -> dict[str, object]:
    import stim

    return {
        "schema_version": "0.1",
        "technology_profile": "v0-1",
        "experiment": {"name": f"cuda-graph-benchmark-{executor}", "master_seed": 2026},
        "qec": {
            "code_family": "toric",
            "distance": 6,
            "rounds": 1,
            "logical_basis": "X",
            "circuit_family": "code-capacity",
        },
        "noise": {
            "family": "independent-phase-flip",
            "adapter": "stim-noise",
            "parameters": {"physical_error_rate": 0.10},
        },
        "dataset": {
            "generator": "stim-syndrome-cpu",
            "generator_version": stim.__version__,
            "backend_semantics": "exact",
            "train_samples": 100_000,
            "validation_samples": 1_000,
            "test_samples": 1_000,
            "seed": 5,
            "split_policy": "independent-streams",
            "schema_version": "qec-batch-v1",
        },
        "model": {
            "family": "joint-error-syndrome-rbm",
            "architecture_version": "1",
            "parameters": {"hidden_units": 128, "init_std": 0.01},
            "decoding": {"burn_in": 10, "max_steps": 300},
        },
        "training": {
            "optimizer": "sgd",
            "optimizer_parameters": {"weight_decay": 0.0, "momentum": 0.0},
            "learning_rate": 0.1,
            "epochs": 3,
            "batch_size": 100,
            "scheduler": "constant",
            "loss": "contrastive-divergence",
            "loss_parameters": {"cd_steps": 10},
        },
        "execution": {
            "trainer_framework": "pytorch",
            "device": "cuda",
            "cpu_count": 1,
            "gpu_count": 1,
            "distributed": False,
            "num_workers": 0,
            "mixed_precision": False,
            "compile_model": False,
            "step_executor": executor,
            "step_executor_options": {"max_graphs": 2} if executor == "pytorch-cuda-graph" else {},
        },
        "data_pipeline": {
            "cpu_to_gpu": {
                "technology": "pytorch-dataloader-h2d",
                "pin_memory": True,
                "non_blocking": True,
            }
        },
        "scientific_evaluation": {
            "baseline_decoders": ["pymatching-cpu-decoder"],
            "primary_metric": "logical_error_rate",
            "confidence_level": 0.95,
            "stopping_rule": "fixed-shots",
            "invalid_sample_policy": "count-as-failure",
        },
        "accuracy_gate": {
            "gate_id": "gate-a",
            "baseline_decoder": "pymatching-cpu-decoder",
            "primary_metric": "logical_error_rate",
            "comparison_rule": "paired-non-inferiority",
            "tolerance": 1.0,
            "confidence_level": 0.95,
        },
        "performance": {"shots": 100, "repetitions": 1, "warmup": 0},
    }


def _stage_benchmark() -> dict[str, object]:
    import torch

    import ai_qec.notebook_api as qec
    from ai_qec.utils.hashing import read_json

    root = Path(tempfile.mkdtemp(prefix="ai-qec-cuda-graph-benchmark-"))
    runtime = qec.LocalNotebookPlatform(root, verbose=False)
    results = {}
    dataset_ids = []
    for executor in ("pytorch-eager", "pytorch-cuda-graph"):
        run = runtime.create_experiment(_config(executor)).start_or_recover()
        dataset = run.resolve_dataset()
        dataset_ids.append(dataset.artifact_id)
        torch.cuda.synchronize()
        started = time.perf_counter()
        model = run.train(dataset)
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - started
        scientific = run.evaluate_accuracy(
            model=model,
            dataset=dataset,
            baselines=("pymatching-cpu-decoder",),
        )
        gate = run.check_accuracy_gate(scientific)
        run.finish()
        metadata = read_json(run.directory / "stages" / "training.meta.json")
        results[executor] = {
            "experiment_id": run.experiment_id,
            "attempt_id": run.attempt_id,
            "dataset_artifact_id": dataset.artifact_id,
            "model_artifact_id": model.artifact_id,
            "model_checksum": model.checksum,
            "training_stage_seconds": training_seconds,
            "step_evidence": metadata["step_executor"],
            "transfer_evidence": metadata["transfer_evidence"],
            "scientific_result_artifact_id": scientific.result_artifact_id,
            "accuracy_gate": gate.decision.value,
        }
    return {
        "root": str(root),
        "same_dataset_artifact": len(set(dataset_ids)) == 1,
        "executors": results,
        "training_stage_speedup": (
            results["pytorch-eager"]["training_stage_seconds"]
            / results["pytorch-cuda-graph"]["training_stage_seconds"]
        ),
    }


def main() -> None:
    import numpy
    import pymatching
    import stim
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to substitute a CPU benchmark")
    result = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": numpy.__version__,
            "stim": stim.__version__,
            "pymatching": pymatching.__version__,
        },
        "microbenchmark": _microbenchmark(args.steps),
        "stage_benchmark": _stage_benchmark(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
