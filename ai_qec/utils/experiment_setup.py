"""Shared setup for experiments launched from notebooks or other Python entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

from ai_qec.qec.codes.base import QECCode
from ai_qec.qec.codes.registry import build_code
from ai_qec.qec.noise.base import NoiseModel
from ai_qec.qec.noise.registry import build_noise_model
from ai_qec.utils.config import first_seed, require_experiment_kind


@dataclass(frozen=True)
class ExperimentSetup:
    device: str
    seed: int
    code: QECCode
    noise: NoiseModel


def find_project_root(start: Path | None = None) -> Path:
    """Locate the source tree from the working directory or an editable install."""
    location = (start or Path.cwd()).resolve()
    source_root = Path(__file__).resolve().parents[2]
    for candidate in (location, *location.parents, source_root):
        # Return early when the compound condition is satisfied.
        if (candidate / "ai_qec" / "__init__.py").is_file() and (candidate / "paper" / "srcs").is_dir():
            return candidate
    raise RuntimeError("请从本项目目录启动 Notebook，或先以 editable 模式安装项目。")


def prepare_run_environment(config: dict[str, Any]) -> tuple[str, int]:
    """Check the requested execution environment and return device and seed."""
    expected_env = config.get("execution", {}).get("conda_env")
    # Reject this state when expected_env and Path(sys.prefix).name != expected_env.
    if expected_env and Path(sys.prefix).name != expected_env:
        raise RuntimeError(f"请选择 {expected_env!r} 内核；当前解释器前缀为 {sys.prefix}")

    device = config.get("training", {}).get("device", "cpu")
    # Reject this state when device not in ('cpu', 'cuda').
    if device not in ("cpu", "cuda"):
        raise ValueError("training.device 必须是 'cpu' 或 'cuda'")
    # Follow this branch when device == 'cuda'.
    if device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("training.device='cuda' 要求安装 PyTorch") from exc
        # Reject this state when not torch.cuda.is_available().
        if not torch.cuda.is_available():
            raise RuntimeError("training.device='cuda' 要求 PyTorch CUDA 设备可用")

    return device, first_seed(config)


def prepare_experiment(config: dict[str, Any]) -> ExperimentSetup:
    """Build the experiment objects with a checked device and seed."""
    device, seed = prepare_run_environment(config)
    return ExperimentSetup(
        device=device,
        seed=seed,
        code=build_code(config),
        noise=build_noise_model(config),
    )



def build_experiment(
    experiment_config: Mapping[str, Any], *, generator: str, model: str,
) -> tuple[QECCode, NoiseModel]:
    """Check the workflow this config selects, then build the objects it describes.

    The code and the noise model are the only objects an entry point derives from the
    experiment parameters alone, so they are built together and handed back as a pair.
    Unlike :func:`prepare_experiment` this takes the experiment parameters on their own,
    before they are merged with a run's reproducibility and execution settings.
    """
    config = dict(experiment_config)
    require_experiment_kind(config, generator=generator, model=model)
    return build_code(config), build_noise_model(config)
