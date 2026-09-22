"""Vendor-neutral training-step executor contracts.

Concrete PyTorch implementations are imported only by
``load_builtin_implementations`` so importing the public notebook facade stays
free of optional runtime dependencies.
"""

from .protocol import (
    BatchSignature,
    BatchSignatureEvidence,
    TensorSignature,
    TrainingStepContext,
    TrainingStepEvidence,
    TrainingStepExecutor,
    TrainingStepPlan,
    TrainingStepResult,
)

__all__ = [
    "BatchSignature",
    "BatchSignatureEvidence",
    "TensorSignature",
    "TrainingStepContext",
    "TrainingStepEvidence",
    "TrainingStepExecutor",
    "TrainingStepPlan",
    "TrainingStepResult",
]
