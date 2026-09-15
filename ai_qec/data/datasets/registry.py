"""Dataset registry."""

from __future__ import annotations

from pathlib import Path

from ai_qec.data.datasets.qec_dataset import QECDataset, load_split


def load_qec_dataset(dataset_dir: str | Path, split: str) -> QECDataset:
    """Load a QEC dataset split through the registry API."""
    return load_split(dataset_dir, split)
