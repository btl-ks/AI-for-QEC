"""Local single-process execution planner; rejects what it cannot honour."""

import os
import platform
import sys

from ai_qec.config_validation import ConfigurationError
from ai_qec.registries import TRAINING_STEP_EXECUTORS
from ai_qec.registry import RegistryError
from ai_qec.training.executors.protocol import TrainingStepPlan
from ai_qec.utils.hashing import sha256_json

from .execution import ExecutionSpec, ResolvedExecutionPlan


class ExecutionConfigurationError(ConfigurationError):
    """The requested execution cannot run here; nothing has been created yet."""


class LocalExecutionPlanner:
    def environment(self, device: str) -> dict[str, object]:
        import numpy
        import torch

        details: dict[str, object] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }
        if device.startswith("cuda"):
            details["device_name"] = torch.cuda.get_device_name(torch.device(device))
        else:
            details["device_name"] = platform.processor() or platform.machine()
        return details

    def resolve(self, spec: ExecutionSpec) -> ResolvedExecutionPlan:
        import torch

        errors: list[str] = []
        if spec.trainer_framework != "pytorch":
            errors.append(
                f"trainer_framework={spec.trainer_framework!r}; the local planner only runs 'pytorch'"
            )
        if spec.distributed:
            errors.append("distributed=True is not supported by the single-process local runtime")
        if spec.mixed_precision:
            errors.append("mixed_precision=True is not supported; RBM training runs in float32")
        if spec.compile_model:
            errors.append("compile_model=True is not supported by the local runtime")
        if spec.num_workers < 0:
            errors.append(f"num_workers must be >= 0, got {spec.num_workers}")
        available_cpus = os.cpu_count() or 1
        if not 1 <= spec.cpu_count <= available_cpus:
            errors.append(f"cpu_count={spec.cpu_count} must be between 1 and {available_cpus}")

        device = spec.device
        if device == "cpu":
            if spec.gpu_count != 0:
                errors.append(f"device='cpu' requires gpu_count=0, got {spec.gpu_count}")
        elif device == "cuda" or device.startswith("cuda:"):
            if not torch.cuda.is_available():
                errors.append(
                    f"device={device!r} requested but CUDA is not available; refusing to fall back to CPU"
                )
            else:
                index = 0 if device == "cuda" else int(device.split(":", 1)[1])
                if index >= torch.cuda.device_count():
                    errors.append(
                        f"device={device!r} but only {torch.cuda.device_count()} CUDA device(s) exist"
                    )
                device = f"cuda:{index}"
            if spec.gpu_count != 1:
                errors.append(f"the local runtime uses exactly one GPU; gpu_count={spec.gpu_count}")
        else:
            errors.append(f"unsupported device {device!r}; use 'cpu', 'cuda' or 'cuda:N'")
        if errors:
            raise ExecutionConfigurationError(
                "unsupported execution configuration:\n  - " + "\n  - ".join(errors)
            )

        try:
            step_plan = TRAINING_STEP_EXECUTORS.build(
                spec.step_executor,
                device=device,
                options=spec.step_executor_options,
            )
        except (ConfigurationError, RegistryError) as error:
            raise ExecutionConfigurationError(str(error)) from error
        if not isinstance(step_plan, TrainingStepPlan):
            raise ExecutionConfigurationError(
                f"[execution.step_executor] {spec.step_executor!r} did not produce a validated plan"
            )

        return ResolvedExecutionPlan(
            device=device,
            world_size=1,
            cpu_count=spec.cpu_count,
            gpu_count=spec.gpu_count,
            num_workers=spec.num_workers,
            distributed_backend=None,
            mixed_precision_mode=None,
            compile_model=False,
            environment_digest=sha256_json(self.environment(device)),
            trainer_framework=spec.trainer_framework,
            step_executor=step_plan.executor_id,
            step_executor_version=step_plan.implementation_version,
            step_executor_options=dict(step_plan.options),
            step_executor_options_digest=step_plan.options_digest,
        )
