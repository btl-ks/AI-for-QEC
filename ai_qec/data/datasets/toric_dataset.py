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
from ai_qec.qec.codes.registry import build_code
from ai_qec.qec.codes.stabilizer import StabilizerCode
from ai_qec.utils.config import config_hash, generation_hash, generation_spec, split_sample_counts


TORIC_DATASET_SCHEMA_VERSION = 1
TORIC_SPLIT_FIELDS = {"split", "dataset_id", "physical_error", "syndrome", "lattice_size", "p_error"}
# Per-split offsets added to the base seed so train/validation/test are independent random streams.
SPLIT_SEED_OFFSETS = {"train": 0, "validation": 10_000, "test": 20_000}


def code_from_manifest(dataset_dir: str | Path) -> StabilizerCode:
    """Build the code a dataset was generated with, as recorded in its manifest.

    The manifest stores the code's name and distance, so a split can be loaded
    without being told which code it belongs to and without widening the on-disk
    schema.
    """
    manifest_path = Path(dataset_dir) / "dataset_manifest.json"
    # Reject this state when not manifest_path.is_file().
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}")
    context = json.loads(manifest_path.read_text(encoding="utf-8")).get("context", {}).get("code", {})
    name, distance = context.get("code"), context.get("distance")
    # Reject this state when not isinstance(name, str) or not isinstance(distance, int).
    if not isinstance(name, str) or not isinstance(distance, int):
        raise ValueError(f"Dataset manifest does not record a code: {manifest_path}")
    return build_code({"qec": {"code": name, "distance": distance}})


@contextmanager
def staged_dataset_dir(output_path: str | Path) -> Iterator[Path]:
    """Yield a staging directory that is atomically renamed to ``output_path`` only on success."""
    target = Path(output_path)
    # Reject this state when target.exists() or target.is_symlink().
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
    # Reject this state when not len(physical_error).
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
    config: dict[str, Any], code: StabilizerCode, files: dict[str, dict[str, Any]], *, extra_context: dict[str, Any] | None = None,
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
        return np.concatenate((self.physical_error, self.syndrome), axis=1).astype(np.float32)


def validate_toric_dataset(dataset_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate immutable raw-error data, hashes, and code-consistent syndromes."""
    root = Path(dataset_dir)
    manifest_path = root / "dataset_manifest.json"
    # Reject this state when not manifest_path.is_file().
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Reject this state when manifest.get('schema_version') != TORIC_DATASET_SCHEMA_VERSION.
    if manifest.get("schema_version") != TORIC_DATASET_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing toric dataset schema_version")
    # Reject this state when manifest.get('representation') != 'error_syndrome'.
    if manifest.get("representation") != "error_syndrome":
        raise ValueError("Dataset is not a raw toric error/syndrome dataset")
    # Reject this state when manifest.get('config_hash') != config_hash(config).
    if manifest.get("config_hash") != config_hash(config):
        raise ValueError("Dataset config hash does not match this run")
    # Reject this state when manifest.get('generation_hash') != generation_hash(config).
    if manifest.get("generation_hash") != generation_hash(config):
        raise ValueError("Dataset generation hash does not match this run")
    # Reject this state when manifest.get('dataset_id') != config['data']['dataset_id'].
    if manifest.get("dataset_id") != config["data"]["dataset_id"]:
        raise ValueError("Dataset ID does not match this run")

    recorded = manifest.get("context", {}).get("code", {})
    size = int(recorded.get("distance", -1))
    # Reject this state when size != int(config['qec']['distance']).
    if size != int(config["qec"]["distance"]):
        raise ValueError("Dataset code distance does not match config")
    # Reject this state when recorded.get('code') != config['qec']['code'].
    if recorded.get("code") != config["qec"]["code"]:
        raise ValueError("Dataset code does not match config")
    code = build_code(config)
    expected_rate = float(config["noise"]["p_error"])
    for split, count in manifest.get("sample_counts", {}).items():
        path = root / f"{split}.npz"
        record = manifest.get("files", {}).get(path.name)
        # Reject this state when the invalid compound condition is detected.
        if not record or not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError(f"Dataset split hash mismatch: {path.name}")
        # Reject this state when int(record.get('samples', -1)) != int(count) or int(count) < 1.
        if int(record.get("samples", -1)) != int(count) or int(count) < 1:
            raise ValueError(f"Dataset split sample count mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as payload:
            # Reject this state when set(payload.files) != TORIC_SPLIT_FIELDS.
            if set(payload.files) != TORIC_SPLIT_FIELDS:
                raise ValueError(f"Toric dataset schema mismatch: {path}")
            errors = payload["physical_error"]
            syndrome = payload["syndrome"]
            # Reject this state when errors.shape != (int(count), code.num_data_qubits).
            if errors.shape != (int(count), code.num_data_qubits):
                raise ValueError(f"Physical-error shape mismatch: {path}")
            # Reject this state when syndrome.shape != (int(count), code.num_syndrome_bits).
            if syndrome.shape != (int(count), code.num_syndrome_bits):
                raise ValueError(f"Syndrome shape mismatch: {path}")
            # Reject this state when the invalid compound condition is detected.
            if not np.isin(errors, (0, 1)).all() or not np.isin(syndrome, (0, 1)).all():
                raise ValueError(f"Non-binary toric data: {path}")
            # Reject this state when not np.array_equal(code.syndrome(errors), syndrome).
            if not np.array_equal(code.syndrome(errors), syndrome):
                raise ValueError(f"Syndrome does not match physical errors: {path}")
            # Reject this state when int(payload['lattice_size'].item()) != size.
            if int(payload["lattice_size"].item()) != size:
                raise ValueError(f"Lattice size metadata mismatch: {path}")
            # Reject this state when not np.isclose(float(payload['p_error'].item()), expected_rate).
            if not np.isclose(float(payload["p_error"].item()), expected_rate):
                raise ValueError(f"Phase-flip probability metadata mismatch: {path}")
    return manifest


def load_toric_split(dataset_dir: str | Path, split: str, code: StabilizerCode | None = None) -> ToricDataset:
    """Load one already validated raw error/syndrome split.

    ``code`` defaults to whatever the dataset's manifest says it was generated with.
    """
    path = Path(dataset_dir) / f"{split}.npz"
    with np.load(path, allow_pickle=False) as payload:
        # Reject this state when set(payload.files) != TORIC_SPLIT_FIELDS.
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
    # Follow this branch when code is None.
    if code is None:
        code = code_from_manifest(dataset_dir)
    # Reject this state when code.distance != dataset.lattice_size.
    if code.distance != dataset.lattice_size:
        raise ValueError(f"Split distance {dataset.lattice_size} does not match code {code.name} d={code.distance}")
    # Reject this state when the invalid compound condition is detected.
    if errors.ndim != 2 or errors.shape[1] != code.num_data_qubits or not len(errors):
        raise ValueError(f"Invalid physical-error shape: {path}")
    # Reject this state when the invalid compound condition is detected.
    if syndrome.shape != (len(errors), code.num_syndrome_bits) or not np.array_equal(code.syndrome(errors), syndrome):
        raise ValueError(f"Invalid syndrome data: {path}")
    return dataset
