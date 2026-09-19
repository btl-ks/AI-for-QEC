"""In-process run lifecycle for notebook entry points: run directory, manifest, and stage records."""

from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import hashlib
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Iterator, Mapping
import uuid

import yaml

from ai_qec.qec.codes.registry import build_code
from ai_qec.qec.noise.registry import build_noise_model
from ai_qec.utils.config import config_hash, resolve_notebook_config, write_json
from ai_qec.utils.experiment_setup import prepare_run_environment
from ai_qec.utils.reproducibility import (
    clean_worktree_error,
    dataset_reference,
    git_diff,
    git_state,
    installed_distributions,
    package_versions,
)
from ai_qec.utils.serialization import format_yaml


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunRecord:
    """Own one immutable ``runs/<run_id>/`` directory and keep its ``run_manifest.json`` current."""

    def __init__(self, config: dict[str, Any], project_root: Path, run_dir: Path, manifest: dict[str, Any]) -> None:
        self.config = config
        self.project_root = project_root
        self.run_dir = run_dir
        self.manifest = manifest

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @classmethod
    def create(
        cls, config: dict[str, Any], config_path: str | Path | None, project_root: str | Path, *, execution: dict[str, Any],
    ) -> "RunRecord":
        """Create a run from a config file or an inline notebook config, saving a YAML snapshot."""
        root = Path(project_root).resolve()
        # Choose the first expression when config_path is not None; otherwise use the fallback.
        config_source = str(Path(config_path).resolve()) if config_path is not None else execution.get("notebook")
        # Reject this state when not isinstance(config_source, str) or not config_source.
        if not isinstance(config_source, str) or not config_source:
            raise ValueError("Inline configuration requires execution.notebook as its source")
        git = git_state(root)
        repro = config["reproducibility"]
        # Follow this branch when repro['require_clean_worktree'].
        if repro["require_clean_worktree"]:
            reason = clean_worktree_error(git)
            # Reject this state when reason.
            if reason:
                raise RuntimeError(f"Refusing to run: {reason}")

        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{stamp}_{config_hash(config)[:12]}_{uuid.uuid4().hex[:8]}"
        run_dir = root / config["outputs"]["runs_root"] / run_id
        run_dir.mkdir(parents=True)
        for name in ("checkpoints", "predictions"):
            (run_dir / name).mkdir()
        # Follow this branch when config_path is None.
        if config_path is None:
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        # Handle all remaining cases.
        else:
            shutil.copy2(config_path, run_dir / "config.yaml")

        environment: dict[str, Any] = {
            "python": sys.version, "executable": sys.executable, "prefix": sys.prefix, "platform": platform.platform(),
        }
        # Follow this branch when repro['save_environment'].
        if repro["save_environment"]:
            environment["packages"] = package_versions()
            environment["distributions_file"] = "environment.json"
            write_json(run_dir / "environment.json", installed_distributions())
        # Follow this branch when repro['save_git_commit'] and git.get('dirty').
        if repro["save_git_commit"] and git.get("dirty"):
            patch = git_diff(root)
            # Follow this branch when patch and patch[0].
            if patch and patch[0]:
                (run_dir / "git_diff.patch").write_bytes(patch[0])
                git["patch_file"] = "git_diff.patch"

        # Choose the first expression when repro['save_git_commit']; otherwise use the fallback.
        # Choose the first expression when config['data']['generator'] == 'toric_code_capacity'; otherwise use the fallback.
        manifest = {
            "schema_version": 1, "run_id": run_id, "experiment": config["experiment"]["name"],
            "config": config_source, "config_hash": config_hash(config), "project_root": str(root),
            "log_file": "run.log",
            "git": git if repro["save_git_commit"] else None,
            "execution": {"python": sys.executable, **execution}, "environment": environment,
            "dataset": dataset_reference(config, root),
            "physics_fidelity": "toric_code_capacity" if config["data"]["generator"] == "toric_code_capacity" else "toy",
            "status": "running", "started_at": utc_now(), "steps": [],
        }
        record = cls(config, root, run_dir, manifest)
        record._log("run started")
        record.save()
        return record

    def _log(self, message: str) -> None:
        with (self.run_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()} {message}\n")

    def save(self) -> None:
        write_json(self.run_dir / "run_manifest.json", self.manifest)

    @contextmanager
    def stage(self, step_id: str) -> Iterator[dict[str, Any]]:
        """Record one stage; append artifact paths to ``record['outputs']`` and they are checked and hashed on success."""
        record: dict[str, Any] = {"id": step_id, "status": "running", "started_at": utc_now(), "outputs": []}
        self.manifest["steps"].append(record)
        self._log(f"stage {step_id} started")
        self.save()
        started = time.perf_counter()
        try:
            yield record
            artifacts = []
            for path in map(Path, record.pop("outputs")):
                # Reject this state when not path.is_file() or path.stat().st_size == 0.
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Stage {step_id} did not create a non-empty artifact: {path}")
                artifacts.append({"path": str(path), "sha256": _sha256(path)})
            record.update(status="success", artifacts=artifacts)
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            self.manifest.update(status="failed", failed_step=step_id, finished_at=utc_now())
            raise
        finally:
            record.update(duration_seconds=time.perf_counter() - started, finished_at=utc_now())
            self._log(f"stage {step_id} {record['status']} in {record['duration_seconds']:.3f}s")
            self.save()

    def refresh_dataset(self) -> None:
        """Re-link the manifest to the dataset now on disk."""
        self.manifest["dataset"] = dataset_reference(self.config, self.project_root)
        self.save()

    def finish(self) -> None:
        # Follow this branch when self.manifest['status'] == 'running'.
        if self.manifest["status"] == "running":
            self.manifest["status"] = "success"
        self.manifest["finished_at"] = utc_now()
        self._log(f"run {self.manifest['status']}")
        self.manifest["log_sha256"] = _sha256(self.run_dir / "run.log")
        self.save()


def start_notebook_run(
    run_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    *,
    project_root: str | Path,
    notebook: str,
    code: Any,
    noise: Any,
) -> tuple[RunRecord, str, int]:
    """Check and show the full effective config before creating a notebook run.

    ``code`` and ``noise`` are the objects the caller built from these same experiment
    parameters.  They are rebuilt here and compared, which catches the notebook failure
    mode of editing the configuration cell and then running only this one; every other
    value is read from the merged config below, so nothing else can go stale.
    """
    parameters = dict(experiment_config)
    # Reject this state when the invalid compound condition is detected.
    if build_code(parameters) != code or build_noise_model(parameters) != noise:
        raise RuntimeError("实验参数已更改；请重新运行实验配置单元格以重建 code 和 noise")

    config = resolve_notebook_config(run_config, experiment_config, project_root=project_root)
    device, seed = prepare_run_environment(config)
    print(f"配置来源：{notebook}；以下是校验后的完整有效配置")
    print(format_yaml(config), end="")
    record = RunRecord.create(
        config, None, project_root,
        execution={"provider": "notebook", "notebook": notebook, "device": device},
    )
    print(f"run 已创建：{record.run_dir}")
    return record, device, seed
