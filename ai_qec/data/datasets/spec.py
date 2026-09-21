"""Declarative dataset generation specification."""

from dataclasses import dataclass

from ai_qec.qec.noise import NoiseSpec
from ai_qec.qec.spec import QECSpec


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """All scientific inputs that may contribute to a DatasetKey."""

    qec: QECSpec
    noise: NoiseSpec
    generator: str
    generator_version: str
    backend_semantics: str
    train_samples: int
    validation_samples: int
    test_samples: int
    seed: int
    split_policy: str
    schema_version: str = "qec-dataset-v1"
    preprocessing: tuple[str, ...] = ()
