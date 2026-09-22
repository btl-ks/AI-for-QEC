"""Local single-process runtime satisfying the paper notebook Protocols.

``LocalNotebookPlatform.create_experiment(config)`` validates everything that
can be validated without side effects (configuration, execution plan, adapter
compatibility, versions) before an Experiment directory is created. Each
``NotebookRun`` method is one recorded Stage; an exception marks the Stage and
Attempt failed (or interrupted for ``KeyboardInterrupt``) and propagates.
``start_or_recover()`` never revives a terminal Attempt: it creates a new one
that reuses only verified completed Stages and resumes training from the newest
verified epoch checkpoint. When the recovery source offers nothing, training,
scientific evaluation and performance also reuse a verified completed Stage of
any Experiment with the same Stage reuse key (``ai_qec.experiment.reuse``);
such Artifacts stay owned by their producer and are read by reference.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import dataclasses
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import time

from ai_qec.config_validation import ConfigurationError, build_from_config, validate_config
from ai_qec.data.datasets.artifact import DatasetArtifact
from ai_qec.data.datasets.local import (
    LocalDatasetResolver,
    LocalDatasetStore,
    code_consistency_validator,
)
from ai_qec.evaluation.performance.throughput import measure_batch_throughput
from ai_qec.evaluation.scientific.local import LocalAccuracyGate, LocalScientificEvaluator
from ai_qec.evaluation.scientific.result import (
    DecoderEvaluation,
    GateDecision,
    MetricEstimate,
    ScientificAcceptanceResult,
    ScientificEvaluationResult,
)
from ai_qec.experiment.artifact import ArtifactKind, ArtifactRef
from ai_qec.experiment.config import (
    REQUIRED_SELECTIONS,
    ExperimentConfiguration,
    parse_experiment_config,
)
from ai_qec.experiment.local import (
    TERMINAL_STATUSES,
    AttemptInProgressError,
    AttemptStateError,
    FileStageRecorder,
    LocalArtifactRepository,
    LocalAttempt,
    LocalExperiment,
    LocalExperimentFactory,
    utc_now,
)
from ai_qec.experiment.recovery import RecoveryPlan, RecoverySource
from ai_qec.experiment.reuse import CROSS_EXPERIMENT_STAGES, PROVENANCE_KEYS, stage_reuse_key
from ai_qec.experiment.run import AttemptStatus
from ai_qec.experiment.stage import StageRecord, StageStatus
from ai_qec.experiment.streams import DerivedRandomStreams
from ai_qec.implementations import load_builtin_implementations
from ai_qec.qec.noise import NoiseApproximation
from ai_qec.registries import DECODERS, LOSSES, OPTIMIZERS
from ai_qec.reporting import figures
from ai_qec.training.checkpoint.model import ModelCheckpoint
from ai_qec.training.checkpoint.recovery import TrainingRecoveryCheckpoint
from ai_qec.training.execution import ResolvedExecutionPlan
from ai_qec.training.execution_planner import LocalExecutionPlanner
from ai_qec.utils.hashing import read_json, to_jsonable, write_json_atomic

STAGE_ORDER = (
    "dataset",
    "training",
    "scientific_evaluation",
    "accuracy_gate",
    "performance",
    "visualization",
)


@dataclass(frozen=True, slots=True)
class PreparedExperiment:
    configuration: ExperimentConfiguration
    plan: ResolvedExecutionPlan
    environment: Mapping[str, object]


@dataclass(slots=True)
class _StageState:
    inputs: list[ArtifactRef]
    outputs: list[ArtifactRef] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True, slots=True)
class _Reuse:
    """A verified completed Stage whose outputs a new Stage adopts instead of recomputing."""

    record: StageRecord
    metadata: dict
    provenance: dict[str, str]

    @property
    def label(self) -> str:
        experiment = self.provenance.get("reused_from_experiment")
        attempt = self.provenance["reused_from"]
        return attempt if experiment is None else f"{experiment}/{attempt}"


def scientific_result_from_json(data: Mapping[str, object]) -> ScientificEvaluationResult:
    return ScientificEvaluationResult(
        evaluation_id=data["evaluation_id"],
        spec_digest=data["spec_digest"],
        dataset_artifact_id=data["dataset_artifact_id"],
        decoder_results=tuple(
            DecoderEvaluation(
                **{**item, "logical_error_rate": MetricEstimate(**item["logical_error_rate"])}
            )
            for item in data["decoder_results"]
        ),
        result_artifact_id=data["result_artifact_id"],
    )


def _checkpoint_ref(checkpoint: ModelCheckpoint) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=checkpoint.artifact_id,
        kind=ArtifactKind.MODEL_CHECKPOINT,
        uri=checkpoint.uri,
        checksum=checkpoint.checksum,
        media_type="application/x-pytorch",
    )


class LocalNotebookPlatform:
    """``NotebookPlatform`` backed by ``datasets/`` and ``runs/`` under a project root."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        datasets_dir: str = "datasets",
        runs_dir: str = "runs",
        verbose: bool = True,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.datasets_root = self.project_root / datasets_dir
        self.runs_root = self.project_root / runs_dir
        self.verbose = verbose
        self.registered = load_builtin_implementations()

    def prepare(self, config: Mapping[str, object]) -> PreparedExperiment:
        """Validate without side effects; every failure happens before any directory exists."""

        import torch

        validate_config(config, required_selections=REQUIRED_SELECTIONS)
        configuration = parse_experiment_config(config)
        spec = configuration.spec
        planner = LocalExecutionPlanner()
        plan = planner.resolve(spec.execution)
        code = build_from_config(config, "qec.code_family", spec=spec.dataset.qec)
        build_from_config(config, "noise.family", spec=spec.dataset.noise)
        compilation = build_from_config(config, "noise.adapter").compile(spec.dataset.noise)
        if compilation.support is not NoiseApproximation.EXACT:
            raise ConfigurationError(
                f"[noise.adapter] cannot represent the requested noise exactly: {compilation.reason}"
            )
        generator = build_from_config(config, "dataset.generator", code=code, noise=compilation)
        compatibility = generator.compatibility(spec.dataset)
        if compatibility.support is not NoiseApproximation.EXACT:
            raise ConfigurationError(f"[dataset.generator] {compatibility.reason}")
        family = build_from_config(config, "model.family")
        family.parameters(spec.model)
        family.decoding_parameters(configuration.decoding)
        LOSSES.build(spec.training.loss, parameters=spec.training.loss_parameters)
        OPTIMIZERS.build(
            spec.training.optimizer,
            parameters=[torch.nn.Parameter(torch.zeros(1))],
            learning_rate=spec.training.learning_rate,
            options=spec.training.optimizer_parameters,
        )
        for baseline in spec.scientific_evaluation.baseline_decoders:
            DECODERS.build(baseline, code=code)
        return PreparedExperiment(configuration, plan, planner.environment(plan.device))

    def create_experiment(self, config: Mapping[str, object]) -> "LocalNotebookExperiment":
        prepared = self.prepare(config)
        factory = LocalExperimentFactory(self.runs_root, prepared.configuration.config)
        experiment = factory.create(prepared.configuration.spec)
        return LocalNotebookExperiment(self, experiment, prepared)

    def artifact_path(self, artifact: ArtifactRef) -> Path:
        """Local file behind an ArtifactRef (URIs are relative to the project root)."""

        return self.project_root / Path(artifact.uri)

    def find_completed_stage(self, stage: str, key: str) -> _Reuse | None:
        """Newest completed ``stage`` of any Experiment with reuse ``key`` whose outputs verify.

        Each candidate's key is recomputed from its Experiment's ``config.json`` and
        the Stage's recorded inputs, so existing runs need no index or migration.
        """

        if not self.runs_root.is_dir():
            return None
        artifacts = LocalArtifactRepository(self.project_root, self.runs_root)
        for experiment_dir in sorted(self.runs_root.iterdir()):
            config_path = experiment_dir / "config.json"
            if not config_path.is_file():
                continue
            config = read_json(config_path)
            stages = FileStageRecorder(experiment_dir)
            for attempt_dir in sorted((experiment_dir / "attempts").glob("attempt-*"), reverse=True):
                record = stages.get(attempt_dir.name, stage)
                if (
                    record is None
                    or record.status is not StageStatus.COMPLETED
                    or record.recovery_metadata_uri is None
                    or not record.output_artifacts
                    or stage_reuse_key(stage, config, record.input_artifacts) != key
                    or not all(artifacts.verify(ref) for ref in record.output_artifacts)
                ):
                    continue
                return _Reuse(
                    record,
                    read_json(artifacts.path(record.recovery_metadata_uri)),
                    {
                        "reused_from": attempt_dir.name,
                        "reused_from_experiment": experiment_dir.name,
                        "reuse_key": key,
                    },
                )
        return None


class LocalNotebookExperiment:
    def __init__(
        self,
        platform: LocalNotebookPlatform,
        experiment: LocalExperiment,
        prepared: PreparedExperiment,
    ):
        self.platform = platform
        self.experiment = experiment
        self.prepared = prepared

    @property
    def experiment_id(self) -> str:
        return self.experiment.experiment_id

    def start_or_recover(self) -> "LocalNotebookRun":
        attempts = self.experiment.attempts()
        if not attempts:
            return LocalNotebookRun(self, self.experiment.start_attempt(), source=None, plan=None)
        latest = attempts[-1]
        if latest.status not in TERMINAL_STATUSES:
            if latest.owner_is_other_live_process():
                raise AttemptInProgressError(
                    f"{self.experiment_id}/{latest.attempt_id} is still owned by a live process: {latest.record['owner']}"
                )
            latest.interrupt(
                "owner process ended, or the same process started this experiment again"
            )
        plan = self._recovery_plan(latest)
        return LocalNotebookRun(self, self.experiment.recover(plan), source=latest, plan=plan)

    def _recovery_plan(self, source: LocalAttempt) -> RecoveryPlan:
        artifacts = LocalArtifactRepository(self.platform.project_root, self.experiment.directory)
        completed: list[ArtifactRef] = []
        resume_from_stage = None
        for stage in STAGE_ORDER:
            record = source.stages.get(source.attempt_id, stage)
            if (
                record is not None
                and record.status is StageStatus.COMPLETED
                and all(artifacts.verify(ref) for ref in record.output_artifacts)
            ):
                completed.extend(record.output_artifacts)
                continue
            resume_from_stage = stage
            break
        training_reusable = resume_from_stage not in ("dataset", "training")
        checkpoint = (
            None if training_reusable else self._latest_recovery_checkpoint(source, artifacts)
        )
        return RecoveryPlan(
            source=RecoverySource(self.experiment_id, source.attempt_id, source.status.value),
            completed_stage_artifacts=tuple(completed),
            training_checkpoint=checkpoint,
            resume_from_stage=resume_from_stage,
        )

    def _latest_recovery_checkpoint(
        self, source: LocalAttempt, artifacts: LocalArtifactRepository
    ) -> TrainingRecoveryCheckpoint | None:
        chain, attempt = set(), source
        while attempt is not None and attempt.attempt_id not in chain:
            chain.add(attempt.attempt_id)
            previous = attempt.recovery_from
            attempt = None if previous is None else self.experiment.attempt(previous["attempt_id"])
        candidates = [
            manifest
            for manifest in artifacts.manifests()
            if manifest.artifact.kind is ArtifactKind.TRAINING_RECOVERY_CHECKPOINT
            and manifest.producer_attempt_id in chain
        ]
        for manifest in sorted(
            candidates, key=lambda item: int(item.metadata["epoch"]), reverse=True
        ):
            if artifacts.verify(manifest.artifact):
                return TrainingRecoveryCheckpoint(
                    checkpoint_id=f"{manifest.producer_attempt_id}.recovery-epoch-{int(manifest.metadata['epoch']):04d}",
                    artifact_id=manifest.artifact.artifact_id,
                    source_attempt_id=manifest.producer_attempt_id,
                    epoch=int(manifest.metadata["epoch"]),
                    global_step=int(manifest.metadata["global_step"]),
                    uri=manifest.artifact.uri,
                    checksum=manifest.artifact.checksum,
                    **manifest.metadata["includes"],
                )
        return None


class LocalNotebookRun:
    def __init__(
        self,
        experiment: LocalNotebookExperiment,
        attempt: LocalAttempt,
        *,
        source: LocalAttempt | None,
        plan: RecoveryPlan | None,
    ) -> None:
        platform = experiment.platform
        self._platform = platform
        self._experiment = experiment.experiment
        self._attempt = attempt
        self._source = source
        self._plan = plan
        self._configuration = experiment.prepared.configuration
        self._config = self._configuration.config
        self._spec = self._configuration.spec
        self._execution_plan = experiment.prepared.plan
        self._environment = experiment.prepared.environment
        self._artifacts = LocalArtifactRepository(platform.project_root, self._experiment.directory)
        self._streams = DerivedRandomStreams(self._spec.master_seed)
        self._code = build_from_config(self._config, "qec.code_family", spec=self._spec.dataset.qec)
        self._family = build_from_config(self._config, "model.family")
        self._store = LocalDatasetStore(platform.datasets_root)
        self._validator = code_consistency_validator(self._code)
        self._dataset: DatasetArtifact | None = None
        self._dataset_ref: ArtifactRef | None = None
        self._training_metadata: dict[str, object] = {}
        self._adopted: dict[str, ArtifactRef] = {}
        self._acceptance: ScientificAcceptanceResult | None = None
        write_json_atomic(
            attempt.directory / "execution_plan.json",
            {"plan": self._execution_plan, "environment": self._environment},
        )
        origin = "" if source is None else f" from {source.attempt_id} ({source.status.value})"
        self._log(
            f"started{origin}; resume stage: {None if plan is None else plan.resume_from_stage}"
        )

    # -- identity -----------------------------------------------------------------------------

    @property
    def attempt_id(self) -> str:
        return self._attempt.attempt_id

    @property
    def experiment_id(self) -> str:
        return self._experiment.experiment_id

    @property
    def directory(self) -> Path:
        return self._attempt.directory

    @property
    def status(self) -> AttemptStatus:
        return self._attempt.status

    # -- stage machinery ----------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self._platform.verbose:
            print(f"[{self.experiment_id} {self.attempt_id}] {message}", flush=True)

    @contextmanager
    def _stage(self, name: str, inputs: Sequence[ArtifactRef] = ()) -> Iterator[_StageState]:
        if self._attempt.status is not AttemptStatus.RUNNING:
            raise AttemptStateError(
                f"{self.attempt_id} is {self._attempt.status.value}; start_or_recover() a new Attempt"
            )
        state = _StageState(inputs=list(inputs))
        started_at, started = utc_now(), time.perf_counter()
        self._attempt.stages.record(
            self.attempt_id,
            StageRecord(
                stage_id=name,
                name=name,
                status=StageStatus.RUNNING,
                started_at=started_at,
                input_artifacts=tuple(inputs),
            ),
        )
        try:
            yield state
        except KeyboardInterrupt:
            self._close_stage(name, started_at, state, StageStatus.INTERRUPTED, "KeyboardInterrupt")
            self._attempt.interrupt(f"{name} interrupted")
            self._log(f"{name}: interrupted")
            raise
        except BaseException as error:
            reason = f"{type(error).__name__}: {error}"
            self._close_stage(name, started_at, state, StageStatus.FAILED, reason)
            self._attempt.fail(f"{name}: {reason}")
            self._log(f"{name}: FAILED ({reason})")
            raise
        self._close_stage(name, started_at, state, StageStatus.COMPLETED, None)
        self._log(f"{name}: {state.summary} [{time.perf_counter() - started:.1f} s]")

    def _close_stage(
        self, name: str, started_at: str, state: _StageState, status: StageStatus, error: str | None
    ) -> None:
        metadata_path = self.directory / "stages" / f"{name}.meta.json"
        write_json_atomic(metadata_path, state.metadata)
        self._attempt.stages.record(
            self.attempt_id,
            StageRecord(
                stage_id=name,
                name=name,
                status=status,
                started_at=started_at,
                finished_at=utc_now(),
                input_artifacts=tuple(state.inputs),
                output_artifacts=tuple(state.outputs),
                error=error,
                recovery_metadata_uri=self._artifacts.uri(metadata_path),
            ),
        )

    def _commit(self, stage: str) -> Callable[..., ArtifactRef]:
        return partial(self._artifacts.commit_file, attempt_id=self.attempt_id, stage_id=stage)

    def _reusable(self, name: str, inputs: Sequence[ArtifactRef]) -> _Reuse | None:
        """Verified outputs of the recovery source, else of any Experiment with the same reuse key."""

        return self._reusable_from_source(name, inputs) or (
            self._platform.find_completed_stage(name, stage_reuse_key(name, self._config, inputs))
            if name in CROSS_EXPERIMENT_STAGES
            else None
        )

    def _reusable_from_source(self, name: str, inputs: Sequence[ArtifactRef]) -> _Reuse | None:
        """Source Stage outputs, if the stage lies before the resume point and still verifies."""

        if self._plan is None or self._source is None:
            return None
        stop = self._plan.resume_from_stage
        if stop is not None and STAGE_ORDER.index(name) >= STAGE_ORDER.index(stop):
            return None
        record = self._source.stages.get(self._source.attempt_id, name)
        if (
            record is None
            or record.status is not StageStatus.COMPLETED
            or tuple(record.input_artifacts) != tuple(inputs)
        ):
            return None
        if not all(self._artifacts.verify(ref) for ref in record.output_artifacts):
            return None
        return _Reuse(
            record,
            read_json(self._artifacts.path(record.recovery_metadata_uri)),
            {"reused_from": self._source.attempt_id},
        )

    def _adopt(self, stage: _StageState, reuse: _Reuse) -> dict:
        """Record the reused outputs and provenance on ``stage``; return the source metadata."""

        stage.outputs.extend(reuse.record.output_artifacts)
        self._adopted.update((ref.artifact_id, ref) for ref in reuse.record.output_artifacts)
        inherited = {key: value for key, value in reuse.metadata.items() if key not in PROVENANCE_KEYS}
        stage.metadata.update(inherited, **reuse.provenance)
        return reuse.metadata

    def _ref(self, artifact_id: str) -> ArtifactRef:
        """An Artifact adopted by this run, else one committed in this Experiment."""

        if artifact_id in self._adopted:
            return self._adopted[artifact_id]
        manifest = self._artifacts.manifest_for(artifact_id)
        if manifest is None:
            raise AttemptStateError(f"unknown artifact {artifact_id}")
        return manifest.artifact

    def _read_json(self, artifact_id: str) -> dict:
        ref = self._ref(artifact_id)
        if not self._artifacts.verify(ref):
            raise AttemptStateError(f"artifact {artifact_id} failed checksum verification")
        return read_json(self._artifacts.path(ref.uri))

    def _stage_outputs(self, stage: str) -> list[ArtifactRef]:
        return [
            manifest.artifact
            for manifest in self._artifacts.manifests()
            if manifest.producer_attempt_id == self.attempt_id
            and manifest.producer_stage_id == stage
        ]

    # -- shared services ----------------------------------------------------------------------

    def _load_split(self, dataset: DatasetArtifact, split: str):
        return self._store.load_split(dataset, split, validator=self._validator)

    def _pipeline(self):
        return build_from_config(
            self._config,
            "data_pipeline.cpu_to_gpu.technology",
            spec=self._configuration.pipeline,
            device=self._execution_plan.device,
        )

    def _require_dataset(self, dataset: DatasetArtifact) -> ArtifactRef:
        if self._dataset is None or dataset.artifact_id != self._dataset.artifact_id:
            raise AttemptStateError(
                "pass the DatasetArtifact returned by this run's resolve_dataset()"
            )
        return self._dataset_ref

    def _require_model(self, model: ModelCheckpoint) -> ArtifactRef:
        ref = _checkpoint_ref(model)
        if not self._artifacts.verify(ref):
            raise AttemptStateError(
                f"model checkpoint {model.checkpoint_id} failed checksum verification"
            )
        return ref

    def _decoders(self, model: ModelCheckpoint) -> dict[str, object]:
        import torch

        payload = torch.load(self._artifacts.path(model.uri), map_location="cpu", weights_only=True)
        if payload.get("model_identity") != model.model_identity:
            raise AttemptStateError(
                f"{model.checkpoint_id} payload identity does not match the ModelCheckpoint"
            )
        network = self._family.load(payload, device=self._execution_plan.device)
        candidate = self._family.build_decoder(
            network,
            code=self._code,
            decoding=self._configuration.decoding,
            pipeline=self._pipeline(),
            device=self._execution_plan.device,
            seed=self._streams.describe("decoding_gibbs").derived_seed,
        )
        baselines = {
            name: DECODERS.build(name, code=self._code)
            for name in self._spec.scientific_evaluation.baseline_decoders
        }
        return {candidate.decoder_id: candidate, **baselines}

    # -- NotebookRun --------------------------------------------------------------------------

    def resolve_dataset(self) -> DatasetArtifact:
        with self._stage("dataset") as stage:
            compilation = build_from_config(self._config, "noise.adapter").compile(
                self._spec.dataset.noise
            )
            resolver = LocalDatasetResolver(
                self._store,
                generator_factory=lambda artifact_id: build_from_config(
                    self._config,
                    "dataset.generator",
                    code=self._code,
                    noise=compilation,
                    dataset_artifact_id=artifact_id,
                ),
                validator=self._validator,
                provenance={
                    "code": self._code.describe(),
                    "noise": {
                        "technology_id": compilation.technology_id,
                        "technology_version": compilation.technology_version,
                        "support": compilation.support.value,
                        "source_spec_digest": compilation.source_spec_digest,
                        "model": repr(compilation.model),
                    },
                },
            )
            resolution = resolver.resolve(self._spec.dataset, self.attempt_id)
            artifact = resolution.artifact
            write_json_atomic(self.directory / "dataset_instance.json", resolution.instance)
            ref = ArtifactRef(
                artifact_id=artifact.artifact_id,
                kind=ArtifactKind.DATASET,
                uri=self._artifacts.uri(self._store.path(artifact.manifest_uri)),
                checksum=artifact.checksum,
                media_type="application/json",
            )
            stage.outputs.append(ref)
            stage.metadata.update(
                dataset_artifact=to_jsonable(artifact),
                instance=to_jsonable(resolution.instance),
                cache_hit=resolution.cache_hit,
            )
            counts = ", ".join(f"{split.name}={split.sample_count}" for split in artifact.splits)
            action = "reused verified" if resolution.cache_hit else "generated and committed"
            stage.summary = f"{action} {artifact.artifact_id} ({counts})"
        self._dataset, self._dataset_ref = artifact, ref
        return artifact

    def train(self, dataset: DatasetArtifact) -> ModelCheckpoint:
        dataset_ref = self._require_dataset(dataset)
        with self._stage("training", inputs=[dataset_ref]) as stage:
            reusable = self._reusable("training", [dataset_ref])
            if reusable is not None:
                metadata = self._adopt(stage, reusable)
                checkpoint = ModelCheckpoint(**metadata["model_checkpoint"])
                stage.summary = f"reused verified model from {reusable.label}"
            else:
                checkpoint = self._train(dataset, stage)
        self._training_metadata = stage.metadata
        return checkpoint

    def _train(self, dataset: DatasetArtifact, stage: _StageState) -> ModelCheckpoint:
        from ai_qec.training.trainers.pytorch_trainer import TrainingContext

        resume = None if self._plan is None else self._plan.training_checkpoint
        epochs = self._spec.training.epochs

        def progress(record: Mapping[str, object]) -> None:
            epoch = int(record["epoch"])
            if epoch == epochs or epoch % 10 == 0:
                self._log(
                    f"training: epoch {epoch}/{epochs} CD objective={record['train_objective']:+.4f} "
                    f"val BCE={record['validation_reconstruction_bce']:.4f} ({record['seconds']:.1f} s/epoch)"
                )

        context = TrainingContext(
            family=self._family,
            load_split=self._load_split,
            pipeline=self._pipeline(),
            streams=self._streams,
            output_dir=self.directory,
            commit_artifact=self._commit("training"),
            resolve_uri=self._artifacts.path,
            resume_from=resume,
            progress=progress,
        )
        trainer = build_from_config(self._config, "execution.trainer_framework", context=context)
        execution = dataclasses.replace(self._spec.execution, device=self._execution_plan.device)
        stage.metadata["step_executor"] = {
            "requested_executor": self._execution_plan.step_executor,
            "observed_executor": None,
            "implementation_version": self._execution_plan.step_executor_version,
            "device": self._execution_plan.device,
            "options_digest": self._execution_plan.step_executor_options_digest,
            "fallback_observed": False,
        }
        try:
            outcome = trainer.train(
                self._spec.model, self._spec.training, execution, dataset, self.attempt_id
            )
        except BaseException as error:
            evidence = getattr(error, "evidence", None)
            if evidence is not None:
                stage.metadata["step_executor"] = to_jsonable(evidence)
            raise
        history_path = self.directory / "training" / "history.json"
        write_json_atomic(history_path, list(outcome.history))
        history_ref = self._commit("training")(
            history_path,
            kind=ArtifactKind.METRICS,
            name="training-history",
            media_type="application/json",
        )
        stage.outputs.extend(self._stage_outputs("training"))
        stage.metadata.update(
            model_checkpoint=to_jsonable(outcome.model_checkpoint),
            history_artifact_id=history_ref.artifact_id,
            resumed_from=to_jsonable(outcome.resumed_from),
            epochs_trained_in_attempt=len(outcome.recovery_checkpoints),
            transfer_evidence=to_jsonable(outcome.transfer_evidence),
            step_executor=to_jsonable(outcome.step_evidence),
        )
        origin = (
            ""
            if resume is None
            else f"resumed after epoch {resume.epoch} from {resume.source_attempt_id}; "
        )
        trained = len(outcome.recovery_checkpoints)
        stage.summary = (
            f"{origin}trained {trained} epoch(s) -> {outcome.model_checkpoint.artifact_id}"
        )
        return outcome.model_checkpoint

    def evaluate_accuracy(
        self,
        model: ModelCheckpoint,
        dataset: DatasetArtifact,
        baselines: Sequence[str],
    ) -> ScientificEvaluationResult:
        declared = self._spec.scientific_evaluation.baseline_decoders
        if tuple(baselines) != declared:
            raise ConfigurationError(
                f"baselines {tuple(baselines)} differ from the pre-registered {declared}"
            )
        inputs = [self._require_dataset(dataset), self._require_model(model)]
        with self._stage("scientific_evaluation", inputs=inputs) as stage:
            reusable = self._reusable("scientific_evaluation", inputs)
            if reusable is not None:
                result = scientific_result_from_json(self._adopt(stage, reusable)["result"])
            else:
                decoders = self._decoders(model)
                evaluator = LocalScientificEvaluator(
                    load_test_batch=lambda artifact: self._load_split(artifact, "test"),
                    candidate_decoder_id=self._family.family_id,
                    protocol={
                        "model.decoding": dict(self._configuration.decoding),
                        "model_identity": model.model_identity,
                    },
                    output_dir=self.directory / "evaluation",
                    commit_artifact=self._commit("scientific_evaluation"),
                )
                result = evaluator.evaluate(
                    self._spec.scientific_evaluation, dataset, decoders, self.attempt_id
                )
                stage.outputs.extend(self._stage_outputs("scientific_evaluation"))
                stage.metadata["result"] = to_jsonable(result)
            stage.summary = "; ".join(
                f"{item.decoder_id} LER={item.logical_error_rate.value:.4f} "
                f"[{item.logical_error_rate.interval_low:.4f}, {item.logical_error_rate.interval_high:.4f}]"
                + (f" timeouts={item.timeout_count}" if item.timeout_count else "")
                for item in result.decoder_results
            ) + ("" if reusable is None else f" (reused from {reusable.label})")
        return result

    def check_accuracy_gate(self, result: ScientificEvaluationResult) -> ScientificAcceptanceResult:
        evidence = self._ref(result.result_artifact_id)
        with self._stage("accuracy_gate", inputs=[evidence]) as stage:
            gate = LocalAccuracyGate(
                read_artifact_json=self._read_json,
                output_dir=self.directory / "gate",
                commit_artifact=self._commit("accuracy_gate"),
            )
            acceptance = gate.assess(self._spec.accuracy_gate, result, self.attempt_id)
            stage.outputs.extend(self._stage_outputs("accuracy_gate"))
            stage.metadata["acceptance"] = to_jsonable(acceptance)
            stage.summary = acceptance.rationale
        self._acceptance = acceptance
        return acceptance

    def evaluate_performance(self, model: ModelCheckpoint, dataset: DatasetArtifact) -> ArtifactRef:
        if self._acceptance is None or self._acceptance.decision is not GateDecision.PASS:
            raise AttemptStateError(
                "performance evaluation requires a PASS from this run's Accuracy Gate"
            )
        inputs = [self._require_dataset(dataset), self._require_model(model)]
        with self._stage("performance", inputs=inputs) as stage:
            reusable = self._reusable("performance", inputs)
            if reusable is not None:
                self._adopt(stage, reusable)
                report = reusable.record.output_artifacts[0]
                stage.summary = f"reused {report.artifact_id} from {reusable.label}"
            else:
                import torch

                settings = self._configuration.performance
                test = self._load_split(dataset, "test")
                shots = min(settings["shots"], len(test.sample_ids))
                batch = dataclasses.replace(
                    test,
                    detector_events=test.detector_events[:shots],
                    observable_truth=test.observable_truth[:shots],
                    physical_errors=test.physical_errors[:shots],
                    sample_ids=test.sample_ids[:shots],
                )
                device = self._execution_plan.device
                synchronize = (
                    (lambda: torch.cuda.synchronize(device))
                    if device.startswith("cuda")
                    else (lambda: None)
                )
                measurement = measure_batch_throughput(
                    self._decoders(model),
                    batch,
                    repetitions=settings["repetitions"],
                    warmup=settings["warmup"],
                    synchronize=synchronize,
                    environment=self._environment,
                )
                path = self.directory / "performance" / "throughput.json"
                write_json_atomic(path, measurement)
                report = self._commit("performance")(
                    path,
                    kind=ArtifactKind.REPORT,
                    name="batch-throughput",
                    media_type="application/json",
                )
                stage.outputs.append(report)
                stage.metadata["report"] = measurement
                stage.summary = "; ".join(
                    f"{name} {entry['shots_per_second']:.0f} shots/s"
                    for name, entry in measurement["decoders"].items()
                )
        return report

    def visualize(
        self,
        scientific_result: ScientificEvaluationResult,
        acceptance: ScientificAcceptanceResult,
        performance_result: ArtifactRef | None,
    ) -> tuple[ArtifactRef, ...]:
        inputs = [
            self._ref(scientific_result.result_artifact_id),
            *(ref for ref in (performance_result,) if ref is not None),
        ]
        with self._stage("visualization", inputs=inputs) as stage:
            directory = self.directory / "figures"
            commit = self._commit("visualization")
            drawn = []
            history_id = self._training_metadata.get("history_artifact_id")
            if history_id is not None:
                drawn.append(
                    (
                        "training-history",
                        figures.plot_training_history(self._read_json(history_id)),
                    )
                )
            drawn.append(
                (
                    "decoder-comparison",
                    figures.plot_decoder_comparison(scientific_result, acceptance),
                )
            )
            drawn.append(("logical-classes", figures.plot_logical_classes(scientific_result)))
            refs = []
            for name, figure in drawn:
                path = figures.save_figure(figure, directory / f"{name}.png")
                refs.append(
                    commit(
                        path,
                        kind=ArtifactKind.REPORT,
                        name=f"figure-{name}",
                        media_type="image/png",
                    )
                )
            stage.outputs.extend(refs)
            stage.summary = f"{len(refs)} figure(s) in {self._artifacts.uri(directory)}"
        return tuple(refs)

    def finish(self) -> None:
        stages = {}
        for name in STAGE_ORDER:
            record = self._attempt.stages.get(self.attempt_id, name)
            stages[name] = "not-run" if record is None else record.status.value
        write_json_atomic(
            self.directory / "summary.json",
            {
                "experiment_id": self.experiment_id,
                "attempt_id": self.attempt_id,
                "stages": stages,
                "gate": None if self._acceptance is None else self._acceptance.decision.value,
                "finished_at": utc_now(),
            },
        )
        self._attempt.complete()
        self._log(f"completed ({', '.join(f'{name}={status}' for name, status in stages.items())})")
