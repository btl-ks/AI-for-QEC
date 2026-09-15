"""Generate immutable raw-error datasets for toric code-capacity experiments."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np

from ai_qec.data.datasets.toric_dataset import TORIC_DATASET_SCHEMA_VERSION, sha256, validate_toric_dataset
from ai_qec.data.schema.manifest import DatasetManifest
from ai_qec.qec.circuits.registry import build_circuit
from ai_qec.qec.codes.registry import build_code
from ai_qec.qec.simulators.registry import build_backend
from ai_qec.qec.noise.registry import build_noise_model
from ai_qec.utils.config import (
    config_hash,
    data_output_dir,
    first_seed,
    generation_hash,
    generation_spec,
    split_sample_counts,
    write_json,
)


def _write_split(
    path: Path,
    *,
    split: str,
    dataset_id: str,
    physical_error: np.ndarray,
    syndrome: np.ndarray,
    lattice_size: int,
    p_error: float,
) -> dict[str, Any]:
    if not len(physical_error):
        raise ValueError("Refusing to write an empty toric dataset split")
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            split=np.asarray(split),
            dataset_id=np.asarray(dataset_id),
            physical_error=physical_error,
            syndrome=syndrome,
            lattice_size=np.asarray(lattice_size),
            p_error=np.asarray(p_error),
        )
    return {
        "sha256": sha256(path),
        "samples": int(len(physical_error)),
        "arrays": {
            "physical_error": {"shape": list(physical_error.shape), "dtype": str(physical_error.dtype)},
            "syndrome": {"shape": list(syndrome.shape), "dtype": str(syndrome.dtype)},
        },
    }


def generate_toric_dataset(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Create or validate a non-empty immutable dataset from the real toric model."""
    output_path = data_output_dir(config, project_root)
    if output_path.exists() or output_path.is_symlink():
        manifest = validate_toric_dataset(output_path, config)
        manifest["reused_immutable"] = True
        return manifest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent / f".staging-{output_path.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    data_cfg = config["data"]
    counts = split_sample_counts(config)
    code = build_code(config)
    circuit = build_circuit(config, code)
    noise = build_noise_model(config)
    backend = build_backend(config, circuit, noise)
    base_seed = first_seed(config)
    batch_size = int(data_cfg["batch_size"])
    split_offsets = {"train": 0, "validation": 10_000, "test": 20_000}

    try:
        files: dict[str, dict[str, Any]] = {}
        for split, n_samples in counts.items():
            rng = np.random.default_rng(base_seed + split_offsets[split])
            errors = np.empty((n_samples, code.num_data_qubits), dtype=np.uint8)
            syndromes = np.empty((n_samples, code.num_syndrome_bits), dtype=np.uint8)
            cursor = 0
            while cursor < n_samples:
                current = min(batch_size, n_samples - cursor)
                batch = backend.sample_batch(current, rng)
                if set(batch) != {"physical_error", "syndrome"}:
                    raise ValueError(f"Toric backend batch keys mismatch: {sorted(batch)}")
                physical_error = np.asarray(batch["physical_error"], dtype=np.uint8)
                syndrome = np.asarray(batch["syndrome"], dtype=np.uint8)
                if physical_error.shape != (current, code.num_data_qubits) or syndrome.shape != (current, code.num_syndrome_bits):
                    raise ValueError("Toric backend batch shape mismatch")
                if not np.array_equal(code.syndrome(physical_error), syndrome):
                    raise ValueError("Toric backend returned inconsistent syndrome data")
                errors[cursor : cursor + current] = physical_error
                syndromes[cursor : cursor + current] = syndrome
                cursor += current
            files[f"{split}.npz"] = _write_split(
                staging / f"{split}.npz",
                split=split,
                dataset_id=str(data_cfg["dataset_id"]),
                physical_error=errors,
                syndrome=syndromes,
                lattice_size=code.distance,
                p_error=float(config["noise"]["p_error"]),
            )

        manifest = DatasetManifest(
            dataset_id=str(data_cfg["dataset_id"]),
            generator=str(data_cfg["generator"]),
            sample_counts=counts,
            feature_names=["physical_error", "syndrome"],
            config_hash=config_hash(config),
            generation_hash=generation_hash(config),
            schema_version=TORIC_DATASET_SCHEMA_VERSION,
            files=files,
            effective_generation=generation_spec(config),
            context={
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "physics_fidelity": "toric_code_capacity",
                "representation": "raw_binary_error_and_syndrome",
                "code": code.context(),
                "circuit": circuit.context(),
                "noise": {"model": noise.name, "p_error": float(config["noise"]["p_error"])},
            },
        ).to_dict()
        manifest["representation"] = "error_syndrome"
        write_json(staging / "dataset_manifest.json", manifest)
        os.rename(staging, output_path)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
