"""Lightweight detector-summary decoder.

The blueprint names a transformer decoder for the D2.2 experiment. The local
environment does not require a deep-learning stack, so V0.1 provides a
drop-in, numpy-only baseline under the transformer package path. It learns from
spatiotemporal detector-summary tokens and saves enough metadata to reproduce
predictions from a checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

def append_bias(features: np.ndarray) -> np.ndarray:
    """Prepend a constant bias column to a 2D feature matrix."""
    return np.column_stack([np.ones(features.shape[0], dtype=features.dtype), features])


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature-wise mean and guarded standard deviation."""
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return mean, scale


def apply_standardizer(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply feature normalization with precomputed mean and scale."""
    return (features - mean) / scale


def _ridge_fit(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Fit ridge regression weights with an unpenalized bias term."""
    x = append_bias(features)
    penalty = alpha * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ target)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically clipped sigmoid transform."""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


@dataclass
class LinearDetectorSummaryDecoder:
    """Numpy baseline that predicts target noise and logical probability."""
    feature_names: list[str]
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    target_weights: np.ndarray | None = None
    logical_weights: np.ndarray | None = None
    ridge_alpha: float = 1e-3

    def fit(
        self,
        features: np.ndarray,
        target_strength: np.ndarray,
        logical_label: np.ndarray,
        ridge_alpha: float = 1e-3,
    ) -> "LinearDetectorSummaryDecoder":
        """Fit target-regression and logical-probability linear heads."""
        self.ridge_alpha = float(ridge_alpha)
        self.feature_mean, self.feature_scale = fit_standardizer(features)
        x = apply_standardizer(features, self.feature_mean, self.feature_scale)
        self.target_weights = _ridge_fit(x, target_strength, self.ridge_alpha)

        centered = logical_label.astype(np.float64)
        centered = np.clip(centered, 1e-4, 1.0 - 1e-4)
        logits = np.log(centered / (1.0 - centered))
        self.logical_weights = _ridge_fit(x, logits, self.ridge_alpha)
        return self

    def _check_fitted(self) -> None:
        """Raise if training parameters required for prediction are missing."""
        # Reject this state when the invalid compound condition is detected.
        if (
            self.feature_mean is None
            or self.feature_scale is None
            or self.target_weights is None
            or self.logical_weights is None
        ):
            raise RuntimeError("Model has not been fitted")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature rows using the training-set standardizer."""
        self._check_fitted()
        assert self.feature_mean is not None
        assert self.feature_scale is not None
        return apply_standardizer(features, self.feature_mean, self.feature_scale)

    def predict_target(self, features: np.ndarray) -> np.ndarray:
        """Predict nonnegative target crosstalk strength."""
        self._check_fitted()
        assert self.target_weights is not None
        predictions = append_bias(self.transform(features)) @ self.target_weights
        return np.clip(predictions, 0.0, None)

    def predict_logical_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict logical-error probability for each sample."""
        self._check_fitted()
        assert self.logical_weights is not None
        logits = append_bias(self.transform(features)) @ self.logical_weights
        return _sigmoid(logits)

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Return all model outputs expected by evaluators and benchmarks."""
        logical_proba = self.predict_logical_proba(features)
        return {
            "target_prediction": self.predict_target(features),
            "logical_probability": logical_proba,
            "logical_prediction": (logical_proba >= 0.5).astype(np.int64),
        }

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Serialize model arrays and optional metadata to an NPZ checkpoint."""
        self._check_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": np.asarray(self.feature_names),
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "target_weights": self.target_weights,
            "logical_weights": self.logical_weights,
            "metadata": np.asarray(json.dumps(metadata or {}, ensure_ascii=False)),
            "ridge_alpha": np.asarray(self.ridge_alpha),
        }
        with target.open("wb") as handle:
            np.savez(handle, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "LinearDetectorSummaryDecoder":
        """Load a model checkpoint produced by save()."""
        return cls.load_with_metadata(path)[0]

    @classmethod
    def load_with_metadata(cls, path: str | Path) -> tuple["LinearDetectorSummaryDecoder", dict[str, Any]]:
        """Load a model and identity metadata, rejecting malformed checkpoints."""
        with np.load(path, allow_pickle=False) as data:
            required = {"feature_names", "feature_mean", "feature_scale", "target_weights", "logical_weights", "metadata", "ridge_alpha"}
            # Reject this state when set(data.files) != required.
            if set(data.files) != required:
                raise ValueError("Checkpoint schema mismatch")
            metadata = json.loads(str(data["metadata"].item()))
            # Reject this state when not isinstance(metadata, dict).
            if not isinstance(metadata, dict):
                raise ValueError("Checkpoint metadata must be an object")
            model = cls(
                feature_names=[str(v) for v in data["feature_names"].tolist()],
                feature_mean=data["feature_mean"].astype(np.float64),
                feature_scale=data["feature_scale"].astype(np.float64),
                target_weights=data["target_weights"].astype(np.float64),
                logical_weights=data["logical_weights"].astype(np.float64),
                ridge_alpha=float(data["ridge_alpha"]),
            )
        return model, metadata


def load_model(path: str | Path) -> LinearDetectorSummaryDecoder:
    """Convenience wrapper for loading exported checkpoints."""
    return LinearDetectorSummaryDecoder.load(path)
