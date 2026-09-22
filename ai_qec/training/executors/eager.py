"""Reference PyTorch eager training-step implementation."""

from collections.abc import Mapping
from types import MappingProxyType

from ai_qec.config_validation import ConfigurationError
from ai_qec.registries import TRAINING_STEP_EXECUTORS
from ai_qec.technology import TechnologyId
from ai_qec.training.executors.protocol import (
    TrainingStepContext,
    TrainingStepEvidence,
    TrainingStepPlan,
    TrainingStepResult,
)
from ai_qec.utils.hashing import sha256_json

EXECUTOR_ID = TechnologyId.PYTORCH_EAGER_STEP.value
IMPLEMENTATION_VERSION = "1"


def _plan(device: str, options: Mapping[str, object]) -> TrainingStepPlan:
    if options:
        raise ConfigurationError(
            "[execution.step_executor_options] pytorch-eager accepts no options; "
            f"got {sorted(options)}"
        )
    normalized: dict[str, object] = {}
    return TrainingStepPlan(
        executor_id=EXECUTOR_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        device=device,
        options=MappingProxyType(normalized),
        options_digest=sha256_json(normalized),
    )


class PyTorchEagerTrainingStepExecutor:
    """The pre-existing five-operation training step behind the common contract."""

    def __init__(self, context: TrainingStepContext, plan: TrainingStepPlan) -> None:
        self.context = context
        self._plan = plan
        self._steps = 0

    @property
    def plan(self) -> TrainingStepPlan:
        return self._plan

    def step(self, tensors: Mapping[str, object]) -> TrainingStepResult:
        visible = self.context.visible(tensors)
        loss = self.context.objective(self.context.model, visible, self.context.generator)
        self.context.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.context.optimizer.step()
        self._steps += 1
        return TrainingStepResult(loss=loss.detach(), execution_mode="eager")

    def finalize(self) -> None:
        return None

    def evidence(self) -> TrainingStepEvidence:
        return TrainingStepEvidence(
            requested_executor=self._plan.executor_id,
            observed_executor=EXECUTOR_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            device=self._plan.device,
            options_digest=self._plan.options_digest,
            signatures=(),
            warmup_steps=0,
            capture_count=0,
            replay_steps=0,
            batch_copy_seconds=0.0,
            capture_seconds=0.0,
            replay_seconds=0.0,
            fallback_observed=False,
        )


@TRAINING_STEP_EXECUTORS.register(EXECUTOR_ID)
def build_pytorch_eager_training_step_executor(
    *,
    device: str,
    options: Mapping[str, object],
    context: TrainingStepContext | None = None,
) -> TrainingStepPlan | PyTorchEagerTrainingStepExecutor:
    plan = _plan(device, options)
    if context is None:
        return plan
    return PyTorchEagerTrainingStepExecutor(context, plan)
