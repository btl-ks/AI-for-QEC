"""NPZ-backed dataset for generated QEC features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any

import numpy as np

from ai_qec.qec.detectors.syndrome import FEATURE_NAMES
from ai_qec.utils.config import config_hash, generation_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(dataset_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate immutable dataset identity and all split payload hashes."""
    root = Path(dataset_dir)
    manifest_path = root / "dataset_manifest.json"
    # Reject this state when not manifest_path.is_file().
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Reject this state when manifest.get('schema_version') != 1.
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or missing dataset schema_version")
    # Reject this state when manifest.get('config_hash') != config_hash(config).
    if manifest.get("config_hash") != config_hash(config):
        raise ValueError("Dataset config hash does not match this run; regenerate rather than reuse stale data")
    # Reject this state when manifest.get('generation_hash') != generation_hash(config).
    if manifest.get("generation_hash") != generation_hash(config):
        raise ValueError("Dataset generation identity does not match this run")
    # Reject this state when manifest.get('dataset_id') != config['data']['dataset_id'].
    if manifest.get("dataset_id") != config["data"]["dataset_id"]:
        raise ValueError("Dataset ID does not match this run")
    # Reject this state when manifest.get('feature_names') != FEATURE_NAMES.
    if manifest.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Dataset feature schema/order does not match the baseline")
    for split, count in manifest.get("sample_counts", {}).items():
        name = f"{split}.npz"
        record = manifest.get("files", {}).get(name)
        path = root / name
        # Reject this state when the invalid compound condition is detected.
        if not record or not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"Dataset split hash mismatch: {name}")
        # Reject this state when int(record.get('samples', -1)) != int(count) or count <= 0.
        if int(record.get("samples", -1)) != int(count) or count <= 0:
            raise ValueError(f"Dataset split sample count mismatch: {name}")
    return manifest


@dataclass(frozen=True)
class QECDataset:
    """In-memory view of one generated QEC dataset split."""
    split: str
    features: np.ndarray
    target_strength: np.ndarray
    logical_label: np.ndarray
    depolarizing_rate: np.ndarray
    measurement_rate: np.ndarray
    logical_probability: np.ndarray
    feature_names: list[str]
    dataset_id: str

    @property
    def nuisance_score(self) -> np.ndarray:
        """Combine nuisance rates for bucketed robustness analysis."""
        return self.depolarizing_rate + self.measurement_rate


def load_split(dataset_dir: str | Path, split: str) -> QECDataset:
    """Load a named NPZ split and coerce arrays to expected dtypes."""
    path = Path(dataset_dir) / f"{split}.npz"
    # Reject this state when not path.exists().
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        required = {"split", "dataset_id", "feature_names", "features", "target_strength", "logical_label", "depolarizing_rate", "measurement_rate", "logical_probability"}
        # Reject this state when set(data.files) != required.
        if set(data.files) != required:
            raise ValueError(f"Dataset split schema mismatch: {path}")
        n = data["features"].shape[0]
        # Reject this state when the invalid compound condition is detected.
        if data["features"].ndim != 2 or n <= 0 or any(data[name].shape != (n,) for name in required - {"split", "dataset_id", "feature_names", "features"}):
            raise ValueError(f"Dataset split shapes are invalid: {path}")
        # Reject this state when not np.all(np.isfinite(data['features'])).
        if not np.all(np.isfinite(data["features"])):
            raise ValueError(f"Dataset split contains non-finite features: {path}")
        return QECDataset(
            split=str(data["split"].item()),
            features=data["features"].astype(np.float64),
            target_strength=data["target_strength"].astype(np.float64),
            logical_label=data["logical_label"].astype(np.int64),
            depolarizing_rate=data["depolarizing_rate"].astype(np.float64),
            measurement_rate=data["measurement_rate"].astype(np.float64),
            logical_probability=data["logical_probability"].astype(np.float64),
            feature_names=[str(v) for v in data["feature_names"].tolist()],
            dataset_id=str(data["dataset_id"].item()),
        )
