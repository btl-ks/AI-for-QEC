"""Shared fixtures: dependency probes and a tiny, fast experiment configuration."""

import copy
import importlib.util
import unittest

RUNTIME_MODULES = ("numpy", "torch", "stim", "pymatching", "matplotlib")
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in RUNTIME_MODULES)

requires_numpy = unittest.skipUnless(HAS_NUMPY, "requires numpy")
requires_runtime = unittest.skipUnless(
    HAS_RUNTIME, "requires the runtime extra: " + ", ".join(RUNTIME_MODULES)
)


def stim_version() -> str:
    import stim

    return stim.__version__


def tiny_config(**overrides: object) -> dict[str, object]:
    """A d=3 CPU experiment that runs end to end in about a second."""

    config: dict[str, object] = {
        "schema_version": "0.1",
        "technology_profile": "v0-1",
        "experiment": {"name": "tiny", "master_seed": 11},
        "qec": {
            "code_family": "toric",
            "distance": 3,
            "rounds": 1,
            "logical_basis": "X",
            "circuit_family": "code-capacity",
        },
        "noise": {
            "family": "independent-phase-flip",
            "adapter": "stim-noise",
            "parameters": {"physical_error_rate": 0.05},
        },
        "dataset": {
            "generator": "stim-syndrome-cpu",
            "generator_version": stim_version() if HAS_RUNTIME else "unknown",
            "backend_semantics": "exact",
            "train_samples": 1_000,
            "validation_samples": 100,
            "test_samples": 200,
            "seed": 5,
            "split_policy": "independent-streams",
            "schema_version": "qec-batch-v1",
        },
        "model": {
            "family": "joint-error-syndrome-rbm",
            "architecture_version": "1",
            "parameters": {"hidden_units": 16, "init_std": 0.01},
            "decoding": {"burn_in": 10, "max_steps": 300},
        },
        "training": {
            "optimizer": "sgd",
            "learning_rate": 0.1,
            "epochs": 3,
            "batch_size": 50,
            "scheduler": "constant",
            "loss": "contrastive-divergence",
            "loss_parameters": {"cd_steps": 2},
        },
        "execution": {
            "trainer_framework": "pytorch",
            "device": "cpu",
            "cpu_count": 1,
            "gpu_count": 0,
            "distributed": False,
            "num_workers": 0,
            "mixed_precision": False,
            "compile_model": False,
        },
        "data_pipeline": {
            "cpu_to_gpu": {
                "technology": "pytorch-dataloader-h2d",
                "pin_memory": True,
                "non_blocking": True,
            },
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
        "performance": {"shots": 50, "repetitions": 1, "warmup": 0},
    }
    config = copy.deepcopy(config)
    for path, value in overrides.items():
        section, key = path.split(".", 1)
        config[section][key] = value
    return config


def run_workflow(runtime, config):
    """The notebook's standard order, returning the run as well."""

    from ai_qec.notebook_api import GateDecision

    run = runtime.create_experiment(config).start_or_recover()
    dataset = run.resolve_dataset()
    model = run.train(dataset)
    scientific = run.evaluate_accuracy(
        model=model,
        dataset=dataset,
        baselines=tuple(config["scientific_evaluation"]["baseline_decoders"]),
    )
    acceptance = run.check_accuracy_gate(scientific)
    performance = None
    if acceptance.decision is GateDecision.PASS:
        performance = run.evaluate_performance(model=model, dataset=dataset)
    figures = run.visualize(
        scientific_result=scientific, acceptance=acceptance, performance_result=performance
    )
    run.finish()
    return run, model, scientific, acceptance, performance, figures
