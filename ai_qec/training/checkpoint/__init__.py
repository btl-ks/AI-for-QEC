"""Distinct model and training-recovery checkpoint contracts."""

from .model import ModelCheckpoint
from .recovery import TrainingRecoveryCheckpoint

__all__ = ["ModelCheckpoint", "TrainingRecoveryCheckpoint"]
