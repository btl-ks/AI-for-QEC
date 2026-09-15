"""Generate synthetic AI-QEC datasets from experiment config."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np

from ai_qec.data.schema.manifest import DatasetManifest
from ai_qec.qec.circuits.registry import build_circuit
from ai_qec.qec.codes.registry import build_code
from ai_qec.qec.detectors.syndrome import FEATURE_NAMES
from ai_qec.qec.noise.registry import build_noise_model
from ai_qec.qec.simulators.registry import build_backend
from ai_qec.utils.config import config_hash, data_output_dir, first_seed, generation_hash, generation_spec, split_sample_counts, write_json


def _empty_arrays(n_samples: int, n_features: int) -> dict[str, np.ndarray]:
    """Allocate all split arrays before filling them in generated batches."""
    return {
        "features": np.empty((n_samples, n_features), dtype=np.float64),
        "target_strength": np.empty(n_samples, dtype=np.float64),
        "logical_label": np.empty(n_samples, dtype=np.int64),
        "depolarizing_rate": np.empty(n_samples, dtype=np.float64),
        "measurement_rate": np.empty(n_samples, dtype=np.float64),
        "logical_probability": np.empty(n_samples, dtype=np.float64),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_split(
    path: Path,
    arrays: dict[str, np.ndarray],
    split: str,
    dataset_id: str,
    feature_names: list[str],
) -> dict[str, Any]:
    """Write one dataset split as an NPZ file with schema metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            split=np.asarray(split),
            dataset_id=np.asarray(dataset_id),
            feature_names=np.asarray(feature_names),
            **arrays,
        )
    with np.load(path, allow_pickle=False) as data:
        files = {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)} for key in data.files}
    return {"sha256": _sha256(path), "arrays": files, "samples": int(arrays["features"].shape[0])}


def _validate_batch(batch: dict[str, Any], n_samples: int, n_features: int) -> None:
    """Reject incomplete or malformed backend output before it reaches disk."""
    expected = {
        "features": ((n_samples, n_features), np.floating),
        "target_strength": ((n_samples,), np.floating),
        "logical_label": ((n_samples,), np.integer),
        "depolarizing_rate": ((n_samples,), np.floating),
        "measurement_rate": ((n_samples,), np.floating),
        "logical_probability": ((n_samples,), np.floating),
    }
    if set(batch) != set(expected):
        raise ValueError(f"Backend batch keys mismatch: expected {sorted(expected)}, got {sorted(batch)}")
    for key, (shape, kind) in expected.items():
        value = np.asarray(batch[key])
        if value.shape != shape or not np.issubdtype(value.dtype, kind) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid backend batch field {key}: shape={value.shape}, dtype={value.dtype}")
    if not np.isin(batch["logical_label"], [0, 1]).all():
        raise ValueError("logical_label must be binary")
    for key in ("target_strength", "depolarizing_rate", "measurement_rate", "logical_probability"):
        if (batch[key] < 0).any() or (batch[key] > 1).any():
            raise ValueError(f"{key} must be in [0, 1]")


def generate_dataset(
    config: dict[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    """Atomically generate immutable train/validation/test NPZ data and manifest."""

    if str(config["data"]["generator"]).lower() == "toric_code_capacity":
        from ai_qec.data.generators.toric_generator import generate_toric_dataset

        return generate_toric_dataset(config, project_root)

    output_path = data_output_dir(config, project_root)
    if output_path.exists() or output_path.is_symlink():
        # Exact immutable content may be reused; a stale/tampered directory is
        # rejected by the same verifier used before train/evaluate.
        from ai_qec.data.datasets.qec_dataset import validate_dataset

        manifest = validate_dataset(output_path, config)
        manifest["reused_immutable"] = True
        return manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent / f".staging-{output_path.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    data_cfg = config["data"]
    dataset_id = str(data_cfg["dataset_id"])
    counts = split_sample_counts(config)

    code = build_code(config)
    circuit = build_circuit(config, code)
    noise = build_noise_model(config)
    backend = build_backend(config, circuit, noise)

    base_seed = first_seed(config)
    batch_size = int(data_cfg.get("batch_size", 8192))
    split_offsets = {"train": 0, "validation": 10_000, "test": 20_000}

    try:
        file_info: dict[str, dict[str, Any]] = {}
        for split, n_samples in counts.items():
            rng = np.random.default_rng(base_seed + split_offsets[split])
            arrays = _empty_arrays(n_samples, len(FEATURE_NAMES))
            cursor = 0
            while cursor < n_samples:
                current = min(batch_size, n_samples - cursor)
                batch = backend.sample_batch(current, rng)
                _validate_batch(batch, current, len(FEATURE_NAMES))
                next_cursor = cursor + current
                for key in arrays:
                    arrays[key][cursor:next_cursor] = batch[key]
                cursor = next_cursor
            file_info[f"{split}.npz"] = _write_split(staging / f"{split}.npz", arrays, split, dataset_id, FEATURE_NAMES)

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            generator=data_cfg["generator"],
            sample_counts=counts,
            feature_names=list(FEATURE_NAMES),
            config_hash=config_hash(config),
            generation_hash=generation_hash(config),
            files=file_info,
            effective_generation=generation_spec(config),
            context={
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "requested_generator": data_cfg["generator"],
                "resolved_backend": backend.name,
                "physics_fidelity": "toy",
                "code": code.context(),
                "circuit": circuit.context(),
            },
        ).to_dict()
        write_json(staging / "dataset_manifest.json", manifest)
        if output_path.exists():
            raise FileExistsError(f"Dataset appeared while generating: {output_path}")
        os.rename(staging, output_path)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
