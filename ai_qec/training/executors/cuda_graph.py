"""Direct PyTorch CUDA Graph executor for a complete optimizer update.

The existing data pipeline still performs host-to-device transfer for every
minibatch.  This executor owns only one fixed-address device buffer per bounded
batch signature and copies the current device minibatch into it before replay.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ai_qec.config_validation import ConfigurationError
from ai_qec.registries import TRAINING_STEP_EXECUTORS
from ai_qec.technology import TechnologyId
from ai_qec.training.executors.protocol import (
    BatchSignature,
    BatchSignatureEvidence,
    TensorSignature,
    TrainingStepContext,
    TrainingStepEvidence,
    TrainingStepPlan,
    TrainingStepResult,
)
from ai_qec.utils.hashing import sha256_json

EXECUTOR_ID = TechnologyId.PYTORCH_CUDA_GRAPH_STEP.value
IMPLEMENTATION_VERSION = "1"
WARMUP_STEPS = 1


class CUDAGraphExecutionError(RuntimeError):
    """Capture or replay failed; the caller must fail the Training Stage."""

    def __init__(self, message: str, evidence: TrainingStepEvidence) -> None:
        super().__init__(message)
        self.evidence = evidence


@dataclass(slots=True)
class _GraphState:
    signature: BatchSignature
    static_inputs: dict[str, object]
    lifecycle: str = "warming"
    graph: object | None = None
    static_loss: object | None = None
    warmup_steps: int = 0
    capture_count: int = 0
    replay_steps: int = 0
    batch_copy_seconds: float = 0.0
    capture_seconds: float = 0.0
    replay_seconds: float = 0.0


def _plan(device: str, options: Mapping[str, object]) -> TrainingStepPlan:
    unknown = sorted(set(options) - {"max_graphs"})
    if unknown:
        raise ConfigurationError(
            f"[execution.step_executor_options] pytorch-cuda-graph unknown options: {unknown}"
        )
    if "max_graphs" not in options:
        raise ConfigurationError(
            "[execution.step_executor_options.max_graphs] is required for pytorch-cuda-graph"
        )
    max_graphs = options["max_graphs"]
    if not isinstance(max_graphs, int) or isinstance(max_graphs, bool) or max_graphs < 1:
        raise ConfigurationError(
            "[execution.step_executor_options.max_graphs] must be a positive integer"
        )
    if not (device == "cuda" or device.startswith("cuda:")):
        raise ConfigurationError(
            "[execution.step_executor] pytorch-cuda-graph requires a CUDA device"
        )
    normalized = {"max_graphs": max_graphs}
    return TrainingStepPlan(
        executor_id=EXECUTOR_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        device=device,
        options=MappingProxyType(normalized),
        options_digest=sha256_json(normalized),
    )


def batch_signature(tensors: Mapping[str, object]) -> BatchSignature:
    """Return the full ordered Tensor signature or reject non-Tensor inputs."""

    import torch

    signatures = []
    for field in sorted(tensors):
        tensor = tensors[field]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"training minibatch field {field!r} is not a Tensor")
        signatures.append(
            TensorSignature(
                field=field,
                shape=tuple(int(value) for value in tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
                stride=tuple(int(value) for value in tensor.stride()),
            )
        )
    if not signatures:
        raise ValueError("training minibatch has no Tensor fields")
    return BatchSignature(tuple(signatures))


class PyTorchCUDAGraphTrainingStepExecutor:
    """Bounded per-signature CUDA Graph cache with no eager fallback."""

    def __init__(self, context: TrainingStepContext, plan: TrainingStepPlan) -> None:
        import torch

        if not torch.cuda.is_available():
            raise ConfigurationError(
                "[execution.step_executor] pytorch-cuda-graph requested but CUDA is unavailable"
            )
        self.context = context
        self._plan = plan
        self._device = torch.device(plan.device)
        self._max_graphs = int(plan.options["max_graphs"])
        self._states: dict[BatchSignature, _GraphState] = {}
        with torch.cuda.device(self._device):
            self._pool = torch.cuda.graph_pool_handle()

    @property
    def plan(self) -> TrainingStepPlan:
        return self._plan

    def _new_state(self, signature: BatchSignature, tensors: Mapping[str, object]) -> _GraphState:
        import torch

        if len(self._states) >= self._max_graphs:
            raise CUDAGraphExecutionError(
                f"batch signature limit max_graphs={self._max_graphs} exceeded before update: "
                f"{signature!r}",
                self.evidence(),
            )
        for item in signature.tensors:
            if torch.device(item.device) != self._device:
                raise CUDAGraphExecutionError(
                    f"batch field {item.field!r} is on {item.device}, expected {self._device}",
                    self.evidence(),
                )
        state = _GraphState(
            signature=signature,
            static_inputs={
                name: torch.empty_like(tensor) for name, tensor in sorted(tensors.items())
            },
        )
        self._states[signature] = state
        return state

    @staticmethod
    def _copy(state: _GraphState, tensors: Mapping[str, object]) -> None:
        started = time.perf_counter()
        for field, target in state.static_inputs.items():
            target.copy_(tensors[field], non_blocking=True)
        state.batch_copy_seconds += time.perf_counter() - started

    def _eager_warmup(self, state: _GraphState) -> TrainingStepResult:
        visible = self.context.visible(state.static_inputs)
        loss = self.context.objective(self.context.model, visible, self.context.generator)
        self.context.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.context.optimizer.step()
        state.warmup_steps += 1
        return TrainingStepResult(loss=loss.detach(), execution_mode="cuda-graph-warmup")

    def _capture(self, state: _GraphState) -> TrainingStepResult:
        import torch

        try:
            with torch.cuda.device(self._device):
                graph = torch.cuda.CUDAGraph()
                version = tuple(
                    int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
                )
                # PyTorch 2.14 captures custom Generator state automatically;
                # older supported versions require explicit graph registration.
                if version < (2, 14):
                    if not hasattr(graph, "register_generator_state"):
                        raise RuntimeError(
                            "this PyTorch build cannot register a custom CUDA generator "
                            "with CUDAGraph"
                        )
                    graph.register_generator_state(self.context.generator)
                torch.cuda.synchronize(self._device)
                started = time.perf_counter()
                with torch.cuda.graph(graph, pool=self._pool):
                    self.context.optimizer.zero_grad(set_to_none=False)
                    loss = self.context.objective(
                        self.context.model,
                        self.context.visible(state.static_inputs),
                        self.context.generator,
                    )
                    loss.backward()
                    self.context.optimizer.step()
                # Stream capture records the update but does not apply it.  The
                # first launch is the one and only update for this capture
                # minibatch; later launches are counted as steady-state replay.
                graph.replay()
                torch.cuda.synchronize(self._device)
                state.capture_seconds += time.perf_counter() - started
        except Exception as error:
            raise CUDAGraphExecutionError(
                f"CUDA Graph capture failed for {state.signature!r}: "
                f"{type(error).__name__}: {error}",
                self.evidence(),
            ) from error
        state.graph = graph
        state.static_loss = loss.detach()
        state.capture_count += 1
        state.lifecycle = "captured"
        return TrainingStepResult(loss=state.static_loss, execution_mode="cuda-graph-capture")

    def _replay(self, state: _GraphState) -> TrainingStepResult:
        import torch

        started = time.perf_counter()
        try:
            with torch.cuda.device(self._device):
                state.graph.replay()
        except Exception as error:
            raise CUDAGraphExecutionError(
                f"CUDA Graph replay failed for {state.signature!r}: "
                f"{type(error).__name__}: {error}",
                self.evidence(),
            ) from error
        state.replay_seconds += time.perf_counter() - started
        state.replay_steps += 1
        return TrainingStepResult(loss=state.static_loss, execution_mode="cuda-graph-replay")

    def step(self, tensors: Mapping[str, object]) -> TrainingStepResult:
        signature = batch_signature(tensors)
        state = self._states.get(signature)
        if state is None:
            state = self._new_state(signature, tensors)
        self._copy(state, tensors)
        if state.warmup_steps < WARMUP_STEPS:
            return self._eager_warmup(state)
        if state.lifecycle != "captured":
            return self._capture(state)
        return self._replay(state)

    def finalize(self) -> None:
        evidence = self.evidence()
        if evidence.capture_count < 1 or evidence.replay_steps < 1:
            raise CUDAGraphExecutionError(
                "pytorch-cuda-graph cannot complete successfully without at least one "
                "capture and one replay",
                evidence,
            )

    def evidence(self) -> TrainingStepEvidence:
        signatures = tuple(
            BatchSignatureEvidence(
                signature=state.signature,
                warmup_steps=state.warmup_steps,
                capture_count=state.capture_count,
                replay_steps=state.replay_steps,
                batch_copy_seconds=state.batch_copy_seconds,
                capture_seconds=state.capture_seconds,
                replay_seconds=state.replay_seconds,
            )
            for state in self._states.values()
        )
        return TrainingStepEvidence(
            requested_executor=self._plan.executor_id,
            observed_executor=EXECUTOR_ID,
            implementation_version=IMPLEMENTATION_VERSION,
            device=self._plan.device,
            options_digest=self._plan.options_digest,
            signatures=signatures,
            warmup_steps=sum(item.warmup_steps for item in signatures),
            capture_count=sum(item.capture_count for item in signatures),
            replay_steps=sum(item.replay_steps for item in signatures),
            batch_copy_seconds=sum(item.batch_copy_seconds for item in signatures),
            capture_seconds=sum(item.capture_seconds for item in signatures),
            replay_seconds=sum(item.replay_seconds for item in signatures),
            fallback_observed=False,
        )


@TRAINING_STEP_EXECUTORS.register(EXECUTOR_ID)
def build_pytorch_cuda_graph_training_step_executor(
    *,
    device: str,
    options: Mapping[str, object],
    context: TrainingStepContext | None = None,
) -> TrainingStepPlan | PyTorchCUDAGraphTrainingStepExecutor:
    plan = _plan(device, options)
    if context is None:
        return plan
    return PyTorchCUDAGraphTrainingStepExecutor(context, plan)
