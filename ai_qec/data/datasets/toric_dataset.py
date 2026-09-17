"""Schema, loading and validation for raw toric-code error-chain datasets."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterator
import uuid

import numpy as np

from ai_qec.data.schema.manifest import DatasetManifest
from ai_qec.qec.codes.toric_code import ToricCode
from ai_qec.utils.config import config_hash, generation_hash, generation_spec, split_sample_counts


TORIC_DATASET_SCHEMA_VERSION = 1
TORIC_SPLIT_FIELDS = {"split", "dataset_id", "physical_error", "syndrome", "lattice_size", "p_error"}
# Per-split offsets added to the base seed so train/validation/test are independent random streams.
SPLIT_SEED_OFFSETS = {"train": 0, "validation": 10_000, "test": 20_000}


@contextmanager
def staged_dataset_dir(output_path: str | Path) -> Iterator[Path]:
    """Yield a staging directory that is atomically renamed to ``output_path`` only on success."""
    target = Path(output_path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable dataset: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".staging-{target.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_toric_split(
    path: Path, *, split: str, dataset_id: str, physical_error: np.ndarray, syndrome: np.ndarray,
    lattice_size: int, p_error: float,
) -> dict[str, Any]:
    """Write one non-empty split and return its manifest file record."""
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


def build_toric_dataset_manifest(
    config: dict[str, Any], code: ToricCode, files: dict[str, dict[str, Any]], *, extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance manifest for split records written by ``write_toric_split``."""
    context = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "physics_fidelity": "toric_code_capacity",
        "representation": "raw_binary_error_and_syndrome",
        "code": code.context(),
        "noise": {"model": str(config["noise"]["model"]), "p_error": float(config["noise"]["p_error"])},
        **(extra_context or {}),
    }
    manifest = DatasetManifest(
        dataset_id=str(config["data"]["dataset_id"]),
        generator=str(config["data"]["generator"]),
        sample_counts=split_sample_counts(config),
        feature_names=["physical_error", "syndrome"],
        config_hash=config_hash(config),
        generation_hash=generation_hash(config),
        schema_version=TORIC_DATASET_SCHEMA_VERSION,
        files=files,
        effective_generation=generation_spec(config),
        context=context,
    ).to_dict()
    manifest["representation"] = "error_syndrome"
    return manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ToricDataset:
    """One validated split of physical Z-error chains and perfect syndromes."""

    split: str
    physical_error: np.ndarray
    syndrome: np.ndarray
    dataset_id: str
    lattice_size: int
    p_error: float

    @property
    def visible(self) -> np.ndarray:
        """Return the paper's RBM visible vector ``[error | syndrome]``."""
        return np.concatenate((self.physical_error, self.syndrome), axis=1).astype(np.float64)


def validate_toric_dataset(dataset_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate immutable raw-error data, hashes, and code-consistent syndromes."""
    root = Path(dataset_dir)
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != TORIC_DATASET_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing toric dataset schema_version")
    if manifest.get("representation") != "error_syndrome":
        raise ValueError("Dataset is not a raw toric error/syndrome dataset")
    if manifest.get("config_hash") != config_hash(config):
        raise ValueError("Dataset config hash does not match this run")
    if manifest.get("generation_hash") != generation_hash(config):
        raise ValueError("Dataset generation hash does not match this run")
    if manifest.get("dataset_id") != config["data"]["dataset_id"]:
        raise ValueError("Dataset ID does not match this run")

    size = int(manifest.get("context", {}).get("code", {}).get("distance", -1))
    if size != int(config["qec"]["distance"]):
        raise ValueError("Dataset lattice size does not match config")
    code = ToricCode(distance=size)
    expected_rate = float(config["noise"]["p_error"])
    for split, count in manifest.get("sample_counts", {}).items():
        path = root / f"{split}.npz"
        record = manifest.get("files", {}).get(path.name)
        if not record or not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError(f"Dataset split hash mismatch: {path.name}")
        if int(record.get("samples", -1)) != int(count) or int(count) < 1:
            raise ValueError(f"Dataset split sample count mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != TORIC_SPLIT_FIELDS:
                raise ValueError(f"Toric dataset schema mismatch: {path}")
            errors = payload["physical_error"]
            syndrome = payload["syndrome"]
            if errors.shape != (int(count), code.num_data_qubits):
                raise ValueError(f"Physical-error shape mismatch: {path}")
            if syndrome.shape != (int(count), code.num_syndrome_bits):
                raise ValueError(f"Syndrome shape mismatch: {path}")
            if not np.isin(errors, (0, 1)).all() or not np.isin(syndrome, (0, 1)).all():
                raise ValueError(f"Non-binary toric data: {path}")
            if not np.array_equal(code.syndrome(errors), syndrome):
                raise ValueError(f"Syndrome does not match physical errors: {path}")
            if int(payload["lattice_size"].item()) != size:
                raise ValueError(f"Lattice size metadata mismatch: {path}")
            if not np.isclose(float(payload["p_error"].item()), expected_rate):
                raise ValueError(f"Phase-flip probability metadata mismatch: {path}")
    return manifest


def load_toric_split(dataset_dir: str | Path, split: str) -> ToricDataset:
    """Load one already validated raw toric-code split."""
    path = Path(dataset_dir) / f"{split}.npz"
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != TORIC_SPLIT_FIELDS:
            raise ValueError(f"Toric dataset schema mismatch: {path}")
        errors = payload["physical_error"].astype(np.uint8)
        syndrome = payload["syndrome"].astype(np.uint8)
        size = int(payload["lattice_size"].item())
        dataset = ToricDataset(
            split=str(payload["split"].item()),
            physical_error=errors,
            syndrome=syndrome,
            dataset_id=str(payload["dataset_id"].item()),
            lattice_size=size,
            p_error=float(payload["p_error"].item()),
        )
    code = ToricCode(distance=dataset.lattice_size)
    if errors.ndim != 2 or errors.shape[1] != code.num_data_qubits or not len(errors):
        raise ValueError(f"Invalid physical-error shape: {path}")
    if syndrome.shape != (len(errors), code.num_syndrome_bits) or not np.array_equal(code.syndrome(errors), syndrome):
        raise ValueError(f"Invalid syndrome data: {path}")
    return dataset
