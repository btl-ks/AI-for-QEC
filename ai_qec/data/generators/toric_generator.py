"""Generate immutable raw-error datasets for toric code-capacity experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_qec.data.datasets.toric_dataset import (
    SPLIT_SEED_OFFSETS,
    build_toric_dataset_manifest,
    staged_dataset_dir,
    validate_toric_dataset,
    write_toric_split,
)
from ai_qec.qec.circuits.registry import build_circuit
from ai_qec.qec.codes.registry import build_code
from ai_qec.qec.simulators.registry import build_backend
from ai_qec.qec.noise.registry import build_noise_model
from ai_qec.utils.config import data_output_dir, first_seed, split_sample_counts, write_json


def generate_toric_dataset(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Create or validate a non-empty immutable dataset from the real toric model."""
    output_path = data_output_dir(config, project_root)
    if output_path.exists() or output_path.is_symlink():
        manifest = validate_toric_dataset(output_path, config)
        manifest["reused_immutable"] = True
        return manifest

    data_cfg = config["data"]
    code = build_code(config)
    circuit = build_circuit(config, code)
    noise = build_noise_model(config)
    backend = build_backend(config, circuit, noise)
    base_seed = first_seed(config)
    batch_size = int(data_cfg["batch_size"])

    files: dict[str, dict[str, Any]] = {}
    with staged_dataset_dir(output_path) as staging:
        for split, n_samples in split_sample_counts(config).items():
            rng = np.random.default_rng(base_seed + SPLIT_SEED_OFFSETS[split])
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
            files[f"{split}.npz"] = write_toric_split(
                staging / f"{split}.npz",
                split=split,
                dataset_id=str(data_cfg["dataset_id"]),
                physical_error=errors,
                syndrome=syndromes,
                lattice_size=code.distance,
                p_error=float(config["noise"]["p_error"]),
            )
        manifest = build_toric_dataset_manifest(config, code, files, extra_context={"circuit": circuit.context()})
        write_json(staging / "dataset_manifest.json", manifest)
    return manifest
