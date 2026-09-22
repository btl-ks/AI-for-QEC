"""PyTorch minibatch trainer selected by ``execution.trainer_framework = pytorch``.

The loop is model-agnostic: the model family builds the network and its input
tensor, ``training.loss`` selects the objective, and ``training.optimizer`` /
``training.scheduler`` select the update rule, all through their Registries.
Every epoch ends with a TrainingRecoveryCheckpoint; the final model is written
separately as a ModelCheckpoint that carries no optimizer or RNG state.
"""

from collections.abc import Callable, Mapping
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import time

import torch

from ai_qec.data.datasets.artifact import DatasetArtifact
from ai_qec.data.pipeline import TransferEvidence
from ai_qec.data.schema.batch import QECBatch
from ai_qec.experiment.artifact import ArtifactKind, ArtifactRef
from ai_qec.models.spec import ModelSpec
from ai_qec.registries import (
    LOSSES,
    OPTIMIZERS,
    SCHEDULERS,
    TRAINERS,
    TRAINING_STEP_EXECUTORS,
)
from ai_qec.training.checkpoint.model import ModelCheckpoint
from ai_qec.training.checkpoint.recovery import TrainingRecoveryCheckpoint
from ai_qec.training.execution import ExecutionSpec
from ai_qec.training.executors.protocol import (
    TrainingStepContext,
    TrainingStepEvidence,
    TrainingStepExecutor,
    TrainingStepPlan,
)
from ai_qec.training.spec import TrainingSpec
from ai_qec.utils.hashing import sha256_file

RECOVERY_SCHEMA = "pytorch-training-recovery-v2"
RECOVERY_CONTENTS = {
    "includes_optimizer": True,
    "includes_scheduler": True,
    "includes_amp_scaler": False,
    "includes_rng_state": True,
    "includes_data_cursor": True,
}


class TrainingRecoveryError(RuntimeError):
    """A recovery checkpoint cannot be used to continue this training run."""


@dataclass(frozen=True, slots=True)
class TrainingContext:
    """Runtime services injected into the trainer by the orchestration layer."""

    family: object
    load_split: Callable[[DatasetArtifact, str], QECBatch]
    pipeline: object
    streams: object
    output_dir: Path
    commit_artifact: Callable[..., ArtifactRef]
    resolve_uri: Callable[[str], Path]
    resume_from: TrainingRecoveryCheckpoint | None = None
    progress: Callable[[Mapping[str, object]], None] | None = None
    monitor_samples: int = 10_000


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    model_checkpoint: ModelCheckpoint
    recovery_checkpoints: tuple[TrainingRecoveryCheckpoint, ...]
    history: tuple[Mapping[str, object], ...]
    transfer_evidence: TransferEvidence
    resumed_from: TrainingRecoveryCheckpoint | None
    step_evidence: TrainingStepEvidence


def _slice(batch: QECBatch, count: int) -> QECBatch:
    return dataclasses.replace(
        batch,
        detector_events=batch.detector_events[:count],
        observable_truth=batch.observable_truth[:count],
        physical_errors=None if batch.physical_errors is None else batch.physical_errors[:count],
        sample_ids=batch.sample_ids[:count],
    )


class PyTorchTrainer:
    def __init__(self, *, context: TrainingContext) -> None:
        self.context = context

    def after_epoch(self, epoch: int, record: Mapping[str, object]) -> None:
        """Called after each epoch's recovery checkpoint is committed."""

        if self.context.progress is not None:
            self.context.progress(record)

    def train(
        self,
        model: ModelSpec,
        training: TrainingSpec,
        execution: ExecutionSpec,
        dataset: DatasetArtifact,
        attempt_id: str,
    ) -> TrainingOutcome:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(execution.cpu_count)
        try:
            return self._train(model, training, execution, dataset, attempt_id)
        finally:
            torch.set_num_threads(previous_threads)

    def _train(
        self,
        model_spec: ModelSpec,
        training: TrainingSpec,
        execution: ExecutionSpec,
        dataset: DatasetArtifact,
        attempt_id: str,
    ) -> TrainingOutcome:
        context = self.context
        family = context.family
        device = torch.device(execution.device)
        train_batch = context.load_split(dataset, "train")
        validation_batch = context.load_split(dataset, "validation")
        for name in family.input_fields:
            if getattr(train_batch, name) is None:
                raise ValueError(f"{family.family_id} requires QECBatch.{name}")
        widths = {name: int(getattr(train_batch, name).shape[1]) for name in family.input_fields}
        identity = family.identity(model_spec, widths)

        network = family.create(
            model_spec, widths=widths, seed=context.streams.describe("model_init").derived_seed
        )
        network.to(device)
        optimizer = OPTIMIZERS.build(
            training.optimizer,
            parameters=network.parameters(),
            learning_rate=training.learning_rate,
            options=training.optimizer_parameters,
        )
        scheduler = SCHEDULERS.build(training.scheduler, optimizer=optimizer)
        objective = LOSSES.build(training.loss, parameters=training.loss_parameters)
        shuffle = torch.Generator().manual_seed(
            context.streams.describe("training_shuffle").derived_seed
        )
        gibbs = torch.Generator(device=device).manual_seed(
            context.streams.describe("training_gibbs").derived_seed
        )
        step_plan = TRAINING_STEP_EXECUTORS.build(
            execution.step_executor,
            device=str(device),
            options=execution.step_executor_options,
        )
        if not isinstance(step_plan, TrainingStepPlan):
            raise TypeError(
                f"training step factory {execution.step_executor!r} did not return a plan"
            )

        monitor = {
            "train": self._visible(_slice(train_batch, context.monitor_samples)),
            "validation": self._visible(validation_batch),
        }
        start_epoch, global_step, history = 1, 0, []
        if context.resume_from is not None:
            start_epoch, global_step, history = self._restore(
                context.resume_from,
                network,
                optimizer,
                scheduler,
                shuffle,
                gibbs,
                identity,
                dataset,
                device,
                step_plan,
            )

        step_executor = TRAINING_STEP_EXECUTORS.build(
            execution.step_executor,
            device=str(device),
            options=execution.step_executor_options,
            context=TrainingStepContext(
                model=network,
                visible=family.visible,
                objective=objective,
                optimizer=optimizer,
                generator=gibbs,
                device=str(device),
            ),
        )
        if not isinstance(step_executor, TrainingStepExecutor):
            raise TypeError(
                f"training step factory {execution.step_executor!r} did not return an executor"
            )

        recovery_checkpoints: list[TrainingRecoveryCheckpoint] = []
        for epoch in range(start_epoch, training.epochs + 1):
            started = time.perf_counter()
            network.train()
            objective_sum = torch.zeros((), device=device)
            batches = 0
            for tensors in context.pipeline.loader(
                train_batch,
                batch_size=training.batch_size,
                generator=shuffle,
                fields=family.input_fields,
            ):
                result = step_executor.step(tensors)
                objective_sum += result.loss.detach()
                batches += 1
                global_step += 1
            scheduler.step()
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_objective": float(objective_sum.item()) / max(batches, 1),
                "train_reconstruction_bce": self._reconstruction(network, monitor["train"]),
                "validation_reconstruction_bce": self._reconstruction(
                    network, monitor["validation"]
                ),
                "seconds": time.perf_counter() - started,
                "attempt_id": attempt_id,
            }
            history.append(record)
            recovery_checkpoints.append(
                self._save_recovery(
                    network,
                    optimizer,
                    scheduler,
                    shuffle,
                    gibbs,
                    epoch,
                    global_step,
                    history,
                    identity,
                    dataset,
                    device,
                    attempt_id,
                    step_plan,
                )
            )
            self.after_epoch(epoch, record)

        step_executor.finalize()
        step_evidence = step_executor.evidence()

        payload = family.payload(
            network, model_spec, widths=widths, dataset_artifact_id=dataset.artifact_id
        )
        path = context.output_dir / "checkpoints" / "model" / "final.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        ref = context.commit_artifact(
            path,
            kind=ArtifactKind.MODEL_CHECKPOINT,
            name="model-final",
            media_type="application/x-pytorch",
            metadata={
                "model_identity": identity,
                "dataset_artifact_id": dataset.artifact_id,
                "epochs": training.epochs,
            },
        )
        model_checkpoint = ModelCheckpoint(
            checkpoint_id=f"{attempt_id}.model-final",
            artifact_id=ref.artifact_id,
            model_identity=identity,
            dataset_artifact_id=dataset.artifact_id,
            uri=ref.uri,
            checksum=ref.checksum,
            selected_metric="final-epoch",
        )
        return TrainingOutcome(
            model_checkpoint=model_checkpoint,
            recovery_checkpoints=tuple(recovery_checkpoints),
            history=tuple(history),
            transfer_evidence=context.pipeline.loader_evidence(train_batch, training.batch_size),
            resumed_from=context.resume_from,
            step_evidence=step_evidence,
        )

    def _visible(self, batch: QECBatch):
        device_batch, _ = self.context.pipeline.transfer(batch)
        fields = {name: getattr(device_batch, name) for name in self.context.family.input_fields}
        return self.context.family.visible(fields)

    @staticmethod
    @torch.no_grad()
    def _reconstruction(network, visible, chunk: int = 65_536) -> float:
        network.eval()
        total = 0.0
        for start in range(0, visible.shape[0], chunk):
            part = visible[start : start + chunk]
            total += float(network.reconstruction_bce(part).item()) * part.shape[0]
        return total / visible.shape[0]

    def _save_recovery(
        self,
        network,
        optimizer,
        scheduler,
        shuffle,
        gibbs,
        epoch,
        global_step,
        history,
        identity,
        dataset,
        device,
        attempt_id,
        step_plan,
    ) -> TrainingRecoveryCheckpoint:
        payload = {
            "schema_version": RECOVERY_SCHEMA,
            "epoch": epoch,
            "global_step": global_step,
            "model_identity": identity,
            "dataset_artifact_id": dataset.artifact_id,
            "device_type": device.type,
            "step_executor": step_plan.executor_id,
            "step_executor_version": step_plan.implementation_version,
            "step_executor_options_digest": step_plan.options_digest,
            "model_state": {
                name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "shuffle_state": shuffle.get_state(),
            "gibbs_state": gibbs.get_state(),
            "history": list(history),
            "source_attempt_id": attempt_id,
        }
        path = self.context.output_dir / "checkpoints" / "recovery" / f"epoch-{epoch:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        ref = self.context.commit_artifact(
            path,
            kind=ArtifactKind.TRAINING_RECOVERY_CHECKPOINT,
            name=f"recovery-epoch-{epoch:04d}",
            media_type="application/x-pytorch",
            metadata={
                "epoch": epoch,
                "global_step": global_step,
                "model_identity": identity,
                "dataset_artifact_id": dataset.artifact_id,
                "step_executor": step_plan.executor_id,
                "step_executor_version": step_plan.implementation_version,
                "step_executor_options_digest": step_plan.options_digest,
                "includes": dict(RECOVERY_CONTENTS),
            },
        )
        return TrainingRecoveryCheckpoint(
            checkpoint_id=f"{attempt_id}.recovery-epoch-{epoch:04d}",
            artifact_id=ref.artifact_id,
            source_attempt_id=attempt_id,
            epoch=epoch,
            global_step=global_step,
            uri=ref.uri,
            checksum=ref.checksum,
            **RECOVERY_CONTENTS,
        )

    def _restore(
        self,
        checkpoint,
        network,
        optimizer,
        scheduler,
        shuffle,
        gibbs,
        identity,
        dataset,
        device,
        step_plan,
    ):
        path = self.context.resolve_uri(checkpoint.uri)
        if not path.is_file() or sha256_file(path) != checkpoint.checksum:
            raise TrainingRecoveryError(
                f"recovery checkpoint {checkpoint.checkpoint_id} failed checksum verification"
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != RECOVERY_SCHEMA:
            raise TrainingRecoveryError(
                f"{checkpoint.checkpoint_id} is not a training recovery checkpoint"
            )
        if (
            payload["model_identity"] != identity
            or payload["dataset_artifact_id"] != dataset.artifact_id
        ):
            raise TrainingRecoveryError(
                f"{checkpoint.checkpoint_id} belongs to a different model or dataset"
            )
        if payload["device_type"] != device.type:
            raise TrainingRecoveryError(
                f"{checkpoint.checkpoint_id} was written on {payload['device_type']}; "
                f"cannot resume on {device.type}"
            )
        observed_executor = (
            payload.get("step_executor"),
            payload.get("step_executor_version"),
            payload.get("step_executor_options_digest"),
        )
        expected_executor = (
            step_plan.executor_id,
            step_plan.implementation_version,
            step_plan.options_digest,
        )
        if observed_executor != expected_executor:
            raise TrainingRecoveryError(
                f"{checkpoint.checkpoint_id} executor metadata {observed_executor!r} "
                f"does not match requested {expected_executor!r}"
            )
        network.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        shuffle.set_state(payload["shuffle_state"])
        gibbs.set_state(payload["gibbs_state"])
        return int(payload["epoch"]) + 1, int(payload["global_step"]), list(payload["history"])


@TRAINERS.register("pytorch")
def build_pytorch_trainer(*, context: TrainingContext) -> PyTorchTrainer:
    return PyTorchTrainer(context=context)
