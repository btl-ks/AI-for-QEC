#!/usr/bin/env python3
"""Execute a resolved AI-QEC experiment with an auditable lifecycle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_qec.utils.config import (  # noqa: E402
    ConfigValidationError,
    ResolvedFlowStep,
    config_hash,
    data_output_dir,
    load_experiment_spec,
)
from ai_qec.utils.reproducibility import (  # noqa: E402
    clean_worktree_error,
    dataset_reference,
    git_diff,
    git_state,
    interpreter_environment,
)

REFUSED_DIRTY_EXIT_CODE = 3
VALIDATION_EXIT_CODE = 2


def _conda_executable() -> str | None:
    """Return the configured Conda executable, if one is available."""
    configured = os.environ.get("CONDA_EXE")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path.resolve())
    found = shutil.which("conda")
    return str(Path(found).resolve()) if found else None


def resolve_execution(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the interpreter and invocation prefix for the selected environment.

    A Conda environment is checked before a run directory is made.  Commands
    remain list-based so the configured environment name is never interpreted by
    a shell.
    """
    execution = config.get("execution")
    if execution is None:
        return {
            "provider": "current",
            # Preserve the virtualenv launcher path; resolving its symlink can
            # escape the environment and make child steps lose dependencies.
            "python": sys.executable,
            "command_prefix": [],
        }

    conda_env = execution["conda_env"]
    conda = _conda_executable()
    if conda is None:
        raise RuntimeError("execution.conda_env is configured but no 'conda' executable is available")
    probe = "import json, sys; print(json.dumps({'executable': sys.executable, 'prefix': sys.prefix}))"
    command = [conda, "run", "--no-capture-output", "-n", conda_env, "python", "-c", probe]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else ""
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"could not start Conda environment '{conda_env}'{detail}") from exc
    try:
        target = json.loads(proc.stdout)
        executable = Path(target["executable"]).resolve()
        prefix = str(target["prefix"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Conda environment '{conda_env}' returned an invalid Python probe") from exc
    if not executable.is_file():
        raise RuntimeError(f"Conda environment '{conda_env}' resolved to a missing Python executable: {executable}")
    return {
        "provider": "conda",
        "conda_env": conda_env,
        "conda_executable": conda,
        "python": str(executable),
        "python_prefix": prefix,
        "command_prefix": [conda, "run", "--no-capture-output", "-n", conda_env],
    }


def expand(value: str, variables: dict[str, str]) -> str:
    for key, replacement in variables.items():
        value = value.replace("${" + key + "}", replacement)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Write a manifest atomically so every observed state is parseable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def check_outputs(step: ResolvedFlowStep, variables: dict[str, str]) -> list[dict[str, Any]]:
    """Verify every declared output contract after a successful child process."""
    records: list[dict[str, Any]] = []
    for contract in step.outputs:
        path = Path(expand(contract.path, variables))
        exists = path.exists() and not path.is_symlink()
        if contract.required and not exists:
            raise RuntimeError(f"Step {step.id} did not create required output: {path}")
        if exists and contract.non_empty:
            if path.is_file() and path.stat().st_size == 0:
                raise RuntimeError(f"Step {step.id} created empty output: {path}")
            if path.is_dir() and not any(path.iterdir()):
                raise RuntimeError(f"Step {step.id} created empty output directory: {path}")
        if exists and contract.schema_version is not None:
            if not path.is_file():
                raise RuntimeError(f"Step {step.id} schema contract requires a file: {path}")
            try:
                schema = json.loads(path.read_text(encoding="utf-8")).get("schema_version")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Step {step.id} schema contract cannot read JSON: {path}") from exc
            if schema != contract.schema_version:
                raise RuntimeError(f"Step {step.id} schema version mismatch at {path}: {schema}")
        record = {"path": str(path), "artifact_type": contract.artifact_type, "required": contract.required}
        if exists and contract.hash_policy == "sha256" and path.is_file():
            record["sha256"] = sha256(path)
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    try:
        spec = load_experiment_spec(config_path, project_root=project_root)
    except (OSError, ConfigValidationError) as exc:
        print(f"Invalid experiment config: {exc}", file=sys.stderr)
        return VALIDATION_EXIT_CODE
    cfg = spec.config
    ids = {step.id for step in spec.flow}
    only = set(args.only) if args.only is not None else None
    skip = set(args.skip)
    unknown = (only or set()).union(skip) - ids
    if unknown:
        print(f"Unknown flow step id(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return VALIDATION_EXIT_CODE
    selected = [step for step in spec.flow if step.enabled and step.id not in skip and (only is None or step.id in only)]
    if not selected:
        print("Refusing zero-step run; adjust --only/--skip or the resolved flow.", file=sys.stderr)
        return VALIDATION_EXIT_CODE
    partial = len(selected) != len([step for step in spec.flow if step.enabled])

    git = git_state(project_root)
    repro = cfg["reproducibility"]
    if repro["require_clean_worktree"] and not args.allow_dirty:
        reason = clean_worktree_error(git)
        if reason:
            print(f"Refusing to run: {reason}\nCommit or stash changes, or pass --allow-dirty.", file=sys.stderr)
            return REFUSED_DIRTY_EXIT_CODE

    try:
        execution = resolve_execution(cfg)
        environment_snapshot = interpreter_environment(execution["python"])
    except RuntimeError as exc:
        print(f"Invalid execution environment: {exc}", file=sys.stderr)
        return VALIDATION_EXIT_CODE
    if cfg["model"]["implementation"] == "joint_error_syndrome_rbm" and not environment_snapshot["packages"].get("torch"):
        print("Invalid execution environment: joint_error_syndrome_rbm requires PyTorch; install ai-qec[torch] in the selected interpreter.", file=sys.stderr)
        return VALIDATION_EXIT_CODE
    if cfg["model"]["implementation"] == "joint_error_syndrome_rbm" and cfg["training"].get("device", "cpu") == "cuda":
        probe = subprocess.run(
            [execution["python"], "-c", "import torch; print(int(torch.cuda.is_available()))"],
            capture_output=True, text=True,
        )
        if probe.returncode or probe.stdout.strip() != "1":
            print("Invalid execution environment: training.device='cuda' requires an available PyTorch CUDA device.", file=sys.stderr)
            return VALIDATION_EXIT_CODE

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{stamp}_{config_hash(cfg)[:12]}_{uuid.uuid4().hex[:8]}"
    run_dir = args.run_dir.resolve() if args.run_dir else project_root / cfg["outputs"]["runs_root"] / run_id
    if run_dir.exists() or run_dir.is_symlink():
        print(f"Refusing existing run directory: {run_dir}", file=sys.stderr)
        return VALIDATION_EXIT_CODE

    variables = {
        "PROJECT_ROOT": str(project_root), "CONFIG": str(config_path), "RUN_DIR": str(run_dir),
        "RUN_ID": run_id, "PYTHON": execution["python"], "DATASET_DIR": str(data_output_dir(cfg, project_root)),
    }
    environment: dict[str, Any] = {
        key: environment_snapshot[key]
        for key in ("python", "executable", "prefix", "platform")
    }
    if repro["save_environment"]:
        environment["packages"] = environment_snapshot["packages"]
        environment["distributions_file"] = "environment.json"
    execution_manifest = {key: value for key, value in execution.items() if key != "command_prefix"}
    manifest: dict[str, Any] = {
        "schema_version": 1, "run_id": run_id, "experiment": cfg["experiment"]["name"],
        "config": str(config_path), "config_hash": config_hash(cfg), "resolved_plan": "resolved_plan.json",
        "project_root": str(project_root), "allow_dirty": args.allow_dirty, "git": git if repro["save_git_commit"] else None,
        "execution": execution_manifest, "environment": environment, "dataset": dataset_reference(cfg, project_root),
        "physics_fidelity": "toric_code_capacity" if cfg["data"]["generator"] == "toric_code_capacity" else "toy",
        "status": "running", "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "steps": [], "partial": partial,
    }
    if args.dry_run:
        for step in selected:
            manifest["steps"].append({"id": step.id, "status": "dry-run"})
        manifest["status"] = "dry-run"
        manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "predictions").mkdir()
    shutil.copy2(config_path, run_dir / "config.yaml")
    atomic_json(run_dir / "resolved_plan.json", spec.to_dict())
    if repro["save_environment"]:
        atomic_json(run_dir / "environment.json", environment_snapshot["distributions"])
    if repro["save_git_commit"] and git.get("dirty"):
        patch = git_diff(project_root)
        if patch and patch[0]:
            (run_dir / "git_diff.patch").write_bytes(patch[0])
            git["patch_file"] = "git_diff.patch"
    atomic_json(run_dir / "run_manifest.json", manifest)

    try:
        for step in spec.flow:
            if not step.enabled:
                manifest["steps"].append({"id": step.id, "status": "disabled", "reason": "config enabled=false"})
                continue
            if step not in selected:
                manifest["steps"].append({"id": step.id, "status": "skipped", "reason": "--only/--skip selection"})
                continue
            command = list(execution["command_prefix"])
            command.extend([execution["python"], str(project_root / step.script)] if step.script else [execution["python"], "-m", str(step.module)])
            command.extend(expand(value, variables) for value in step.args)
            record: dict[str, Any] = {"id": step.id, "command": command, "status": "running", "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
            manifest["steps"].append(record)
            atomic_json(run_dir / "run_manifest.json", manifest)
            log = run_dir / "logs" / f"{step.id}.log"
            started = time.perf_counter()
            with log.open("w", encoding="utf-8") as handle:
                proc = subprocess.run(command, cwd=project_root, stdout=handle, stderr=subprocess.STDOUT, text=True)
            record.update({"returncode": proc.returncode, "log": str(log), "duration_seconds": time.perf_counter() - started, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            if proc.returncode:
                record["status"] = "failed"
                manifest.update(status="failed", failed_step=step.id, finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
                atomic_json(run_dir / "run_manifest.json", manifest)
                print(f"Step failed: {step.id}. See {log}", file=sys.stderr)
                return proc.returncode
            record["artifacts"] = check_outputs(step, variables)
            record["status"] = "success"
            atomic_json(run_dir / "run_manifest.json", manifest)
    except KeyboardInterrupt:
        manifest.update(status="interrupted", finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(run_dir / "run_manifest.json", manifest)
        return 130
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(run_dir / "run_manifest.json", manifest)
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    manifest["dataset"] = dataset_reference(cfg, project_root)
    manifest.update(status="partial" if partial else "success", finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(run_dir / "run_manifest.json", manifest)
    print(f"Run completed: {run_dir} ({manifest['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
