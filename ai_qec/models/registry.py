"""Model registry."""

from __future__ import annotations

from typing import Any

from ai_qec.models.decoders.transformer.model import LinearDetectorSummaryDecoder


def build_model(
    config: dict[str, Any],
    feature_names: list[str] | None = None,
    *,
    error_units: int | None = None,
    syndrome_units: int | None = None,
) -> Any:
    """Build the configured decoder implementation for the given features."""
    implementation = config["model"]["implementation"]
    # Follow this branch when implementation == 'linear_detector_summary_baseline'.
    if implementation == "linear_detector_summary_baseline":
        # Reject this state when feature_names is None.
        if feature_names is None:
            raise ValueError("linear decoder requires feature_names")
        return LinearDetectorSummaryDecoder(feature_names=feature_names)
    # Follow this branch when implementation == 'joint_error_syndrome_rbm'.
    if implementation == "joint_error_syndrome_rbm":
        from ai_qec.models.decoders.generative.rbm import JointErrorSyndromeRBM
        # Reject this state when error_units is None or syndrome_units is None.
        if error_units is None or syndrome_units is None:
            raise ValueError("RBM requires error_units and syndrome_units")
        return JointErrorSyndromeRBM(
            error_units=error_units,
            syndrome_units=syndrome_units,
            hidden_units=int(config["model"]["hidden_units"]),
            init_width=float(config["model"]["init_width"]),
            seed=int(config["reproducibility"]["seeds"][0]),
        )
    raise ValueError(f"Unknown model implementation: {implementation}")


def load_model(config: dict[str, Any], checkpoint: str) -> tuple[Any, dict[str, Any]]:
    """Load the configured model through the same implementation registry."""
    implementation = config["model"]["implementation"]
    # Follow this branch when implementation == 'joint_error_syndrome_rbm'.
    if implementation == "joint_error_syndrome_rbm":
        from ai_qec.models.decoders.generative.rbm import JointErrorSyndromeRBM
        return JointErrorSyndromeRBM.load_with_metadata(checkpoint)
    raise ValueError(f"Checkpoint loading is not registered for {implementation}")
