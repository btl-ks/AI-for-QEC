"""File-backed Experiment, Attempt, Stage and Artifact persistence (single writer).

Layout under the runs root::

    <experiment-id>/experiment.json, config.json
    <experiment-id>/artifacts/<artifact-id>.json          ArtifactManifest, written once
    <experiment-id>/attempts/attempt-NNNN/attempt.json    status, owner, recovery_from
    <experiment-id>/attempts/attempt-NNNN/stages/<stage>.json

Artifact URIs are POSIX paths relative to the project root.
"""

from collections.abc import Mapping
from datetime import datetime, UTC
import os
from pathlib import Path
import socket

from ai_qec.utils.hashing import read_json, sha256_file, sha256_json, to_jsonable, write_json_atomic

from .artifact import ArtifactKind, ArtifactManifest, ArtifactRef
from .recovery import RecoveryPlan
from .run import AttemptStatus
from .spec import ExperimentSpec
from .stage import StageRecord, StageStatus

TERMINAL_STATUSES = frozenset(
    {
        AttemptStatus.COMPLETED,
        AttemptStatus.FAILED,
        AttemptStatus.INTERRUPTED,
        AttemptStatus.PARTIAL,
    }
)


class AttemptStateError(RuntimeError):
    """An illegal Attempt transition, such as reviving a terminal Attempt."""


class AttemptInProgressError(AttemptStateError):
    """Another live process still owns the latest Attempt."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def artifact_ref_from_json(data: Mapping[str, object]) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=str(data["artifact_id"]),
        kind=ArtifactKind(data["kind"]),
        uri=str(data["uri"]),
        checksum=str(data["checksum"]),
        media_type=str(data["media_type"]),
    )


class LocalArtifactRepository:
    """ArtifactRepository whose manifests live inside one experiment directory."""

    def __init__(self, project_root: Path, experiment_dir: Path) -> None:
        self.project_root = Path(project_root)
        self.manifests_dir = Path(experiment_dir) / "artifacts"

    def path(self, uri: str) -> Path:
        return self.project_root / Path(uri)

    def uri(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.project_root.resolve()).as_posix()

    def commit_file(
        self,
        path: Path,
        *,
        kind: ArtifactKind,
        name: str,
        attempt_id: str,
        stage_id: str,
        media_type: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef:
        ref = ArtifactRef(
            artifact_id=f"{attempt_id}.{name}",
            kind=kind,
            uri=self.uri(path),
            checksum=sha256_file(path),
            media_type=media_type,
        )
        manifest = ArtifactManifest(
            artifact=ref,
            producer_attempt_id=attempt_id,
            producer_stage_id=stage_id,
            created_at=utc_now(),
            metadata=dict(metadata or {}),
        )
        return self.commit(manifest)

    def commit(self, manifest: ArtifactManifest) -> ArtifactRef:
        target = self.manifests_dir / f"{manifest.artifact.artifact_id}.json"
        if target.exists():
            existing = self.manifest_for(manifest.artifact.artifact_id)
            if existing is None or existing.artifact != manifest.artifact:
                raise AttemptStateError(
                    f"artifact {manifest.artifact.artifact_id} is already committed"
                )
            return existing.artifact
        write_json_atomic(target, manifest)
        return manifest.artifact

    def manifest_for(self, artifact_id: str) -> ArtifactManifest | None:
        target = self.manifests_dir / f"{artifact_id}.json"
        if not target.is_file():
            return None
        data = read_json(target)
        return ArtifactManifest(
            artifact=artifact_ref_from_json(data["artifact"]),
            producer_attempt_id=data["producer_attempt_id"],
            producer_stage_id=data["producer_stage_id"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
            schema_version=data["schema_version"],
        )

    def manifests(self) -> list[ArtifactManifest]:
        if not self.manifests_dir.is_dir():
            return []
        return [self.manifest_for(path.stem) for path in sorted(self.manifests_dir.glob("*.json"))]

    def verify(self, artifact: ArtifactRef) -> bool:
        path = self.path(artifact.uri)
        return path.is_file() and sha256_file(path) == artifact.checksum

    def read_json(self, artifact_id: str) -> dict:
        manifest = self.manifest_for(artifact_id)
        if manifest is None:
            raise KeyError(f"unknown artifact {artifact_id}")
        if not self.verify(manifest.artifact):
            raise AttemptStateError(f"artifact {artifact_id} failed checksum verification")
        return read_json(self.path(manifest.artifact.uri))


class FileStageRecorder:
    def __init__(self, experiment_dir: Path) -> None:
        self.experiment_dir = Path(experiment_dir)

    def _path(self, attempt_id: str, stage_id: str) -> Path:
        return self.experiment_dir / "attempts" / attempt_id / "stages" / f"{stage_id}.json"

    def record(self, attempt_id: str, stage: StageRecord) -> None:
        write_json_atomic(self._path(attempt_id, stage.stage_id), stage)

    def get(self, attempt_id: str, stage_id: str) -> StageRecord | None:
        path = self._path(attempt_id, stage_id)
        if not path.is_file():
            return None
        data = read_json(path)
        return StageRecord(
            stage_id=data["stage_id"],
            name=data["name"],
            status=StageStatus(data["status"]),
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            input_artifacts=tuple(artifact_ref_from_json(item) for item in data["input_artifacts"]),
            output_artifacts=tuple(
                artifact_ref_from_json(item) for item in data["output_artifacts"]
            ),
            error=data["error"],
            recovery_metadata_uri=data["recovery_metadata_uri"],
        )


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LocalAttempt:
    """One Attempt; terminal states are final and persisted immediately."""

    def __init__(self, experiment_dir: Path, attempt_id: str) -> None:
        self.directory = Path(experiment_dir) / "attempts" / attempt_id
        self._attempt_id = attempt_id
        self._stages = FileStageRecorder(experiment_dir)

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def record(self) -> dict:
        return read_json(self.directory / "attempt.json")

    @property
    def status(self) -> AttemptStatus:
        return AttemptStatus(self.record["status"])

    @property
    def stages(self) -> FileStageRecorder:
        return self._stages

    @property
    def recovery_from(self) -> dict | None:
        return self.record.get("recovery_from")

    def _write(self, record: Mapping[str, object]) -> None:
        write_json_atomic(self.directory / "attempt.json", record)

    def _transition(self, status: AttemptStatus, reason: str | None) -> None:
        record = self.record
        current = AttemptStatus(record["status"])
        if current in TERMINAL_STATUSES:
            raise AttemptStateError(
                f"{self.attempt_id} is already {current.value}; terminal Attempts are immutable"
            )
        record.update(status=status.value, reason=reason, finished_at=utc_now())
        self._write(record)

    def complete(self) -> None:
        self._transition(AttemptStatus.COMPLETED, None)

    def fail(self, reason: str) -> None:
        self._transition(AttemptStatus.FAILED, reason)

    def interrupt(self, reason: str) -> None:
        self._transition(AttemptStatus.INTERRUPTED, reason)

    def owner_is_other_live_process(self) -> bool:
        owner = self.record.get("owner", {})
        if owner.get("host") != socket.gethostname():
            return True
        pid = int(owner.get("pid", -1))
        return pid != os.getpid() and _process_alive(pid)


class LocalExperiment:
    def __init__(self, runs_root: Path, experiment_id: str, spec: ExperimentSpec) -> None:
        self.directory = Path(runs_root) / experiment_id
        self._experiment_id = experiment_id
        self._spec = spec

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def spec(self) -> ExperimentSpec:
        return self._spec

    def attempts(self) -> list[LocalAttempt]:
        root = self.directory / "attempts"
        if not root.is_dir():
            return []
        return [
            LocalAttempt(self.directory, path.name)
            for path in sorted(root.glob("attempt-*"))
            if (path / "attempt.json").is_file()
        ]

    def attempt(self, attempt_id: str) -> LocalAttempt:
        attempt = LocalAttempt(self.directory, attempt_id)
        if not (attempt.directory / "attempt.json").is_file():
            raise KeyError(f"{self.experiment_id} has no attempt {attempt_id}")
        return attempt

    def _new_attempt(self, recovery: Mapping[str, object] | None) -> LocalAttempt:
        root = self.directory / "attempts"
        existing = (
            [int(path.name.split("-", 1)[1]) for path in root.glob("attempt-*")]
            if root.is_dir()
            else []
        )
        attempt_id = f"attempt-{max(existing, default=0) + 1:04d}"
        attempt = LocalAttempt(self.directory, attempt_id)
        attempt.directory.mkdir(parents=True, exist_ok=False)
        attempt._write(
            {
                "attempt_id": attempt_id,
                "experiment_id": self.experiment_id,
                "status": AttemptStatus.RUNNING.value,
                "created_at": utc_now(),
                "finished_at": None,
                "reason": None,
                "owner": {"pid": os.getpid(), "host": socket.gethostname()},
                "recovery_from": None if recovery is None else recovery["source"],
                "recovery_plan": None if recovery is None else recovery,
            }
        )
        return attempt

    def start_attempt(self) -> LocalAttempt:
        return self._new_attempt(None)

    def recover(self, plan: RecoveryPlan) -> LocalAttempt:
        source = self.attempt(plan.source.source_attempt_id)
        if source.status not in TERMINAL_STATUSES:
            raise AttemptStateError(
                f"cannot recover from non-terminal {source.attempt_id} ({source.status.value})"
            )
        return self._new_attempt(
            {
                "source": {
                    "attempt_id": plan.source.source_attempt_id,
                    "terminal_status": plan.source.source_terminal_status,
                },
                "completed_stage_artifacts": [
                    to_jsonable(ref) for ref in plan.completed_stage_artifacts
                ],
                "training_checkpoint": to_jsonable(plan.training_checkpoint),
                "resume_from_stage": plan.resume_from_stage,
            }
        )


class LocalExperimentFactory:
    """Experiment identity is the canonical digest of the full configuration."""

    def __init__(self, runs_root: Path, config: Mapping[str, object]) -> None:
        self.runs_root = Path(runs_root)
        self.config = config
        self.config_digest = sha256_json(config)

    def experiment_id(self, spec: ExperimentSpec) -> str:
        return f"{spec.experiment_name}-{self.config_digest.split(':', 1)[1][:12]}"

    def create(self, spec: ExperimentSpec) -> LocalExperiment:
        experiment = LocalExperiment(self.runs_root, self.experiment_id(spec), spec)
        record_path = experiment.directory / "experiment.json"
        if record_path.is_file():
            if read_json(record_path)["config_digest"] != self.config_digest:
                raise AttemptStateError(
                    f"{experiment.experiment_id} exists with a different configuration digest"
                )
            return experiment
        write_json_atomic(experiment.directory / "config.json", self.config)
        write_json_atomic(
            record_path,
            {
                "experiment_id": experiment.experiment_id,
                "experiment_name": spec.experiment_name,
                "config_digest": self.config_digest,
                "spec": to_jsonable(spec),
                "created_at": utc_now(),
                "schema_version": "local-experiment-v1",
            },
        )
        return experiment
