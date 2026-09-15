"""Schema, loading and validation for raw toric-code error-chain datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ai_qec.qec.codes.toric_code import ToricCode
from ai_qec.utils.config import config_hash, generation_hash


TORIC_DATASET_SCHEMA_VERSION = 1
TORIC_SPLIT_FIELDS = {"split", "dataset_id", "physical_error", "syndrome", "lattice_size", "p_error"}


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
