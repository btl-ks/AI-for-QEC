#!/usr/bin/env python3
"""
Export a minimal reproducible paper artifact from an AI-QEC experiment.

设计目标：
- 只复制 YAML 中显式 allowlist 的文件。
- 每个内部路径显式映射到外部 release 路径。
- 不递归暴露整个项目结构。
- 自动生成最小公开 config、文件 hash 清单和 zip。
- 支持 ${PROJECT_ROOT}, ${RUN_DIR}, ${RUN_ID}, ${CONFIG} 变量。

发布安全：
- 输出只能是 <project>/paper/releases/ 下的包目录；拒绝项目根及其祖先、run/data 根、
  符号链接逃逸。导出器从不删除或覆盖任何既有 release。
- 每次导出创建新的不可变 release ID；先在包目录内的 sibling staging 目录完整构建并
  校验，再原子 rename 提交，最后原子替换 LATEST 指针。任何失败或中断都保持旧 release
  与旧指针不变。

发布布局：
    paper/releases/<package>/        # --output，默认 paper/releases/<package_name>
    ├── LATEST                       # 最新 release ID
    └── <release_id>/                # 不可变
        ├── <package>/               # 复现包内容，含 MANIFEST.sha256.json
        └── <package>.zip            # 同内容压缩包（--no-zip 时省略）

Usage:
    python scripts/export_paper.py \
        --config configs/experiment/example.yaml \
        --run-dir runs/<run_id>

依赖：
    pip install pyyaml
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_qec.utils.config import ConfigValidationError, config_hash, load_config  # noqa: E402

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

EXPORT_ERROR_EXIT_CODE = 2
LATEST_POINTER = "LATEST"
MANIFEST_NAME = "MANIFEST.sha256.json"
METADATA_NAME = "artifact_metadata.json"
RELEASE_INDEX_NAME = "RELEASE_INDEX.json"
STAGING_PREFIX = ".staging-"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ExportError(Exception):
    """Raised when an export request is unsafe or the built release fails verification."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Load an artifact YAML file and require a top-level mapping."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping.")
    return data


def validate_run_closure(cfg: dict[str, Any], run_dir: Path | None) -> None:
    """Require a successful, identity-consistent run before publishing a package."""
    if run_dir is None:
        raise ExportError("--run-dir is required: paper packages may not be built without a completed run")
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ExportError(f"Run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        raise ExportError("Only a complete successful run can be exported")
    if manifest.get("config_hash") != config_hash(cfg):
        raise ExportError("Run config hash does not match export config")
    dataset = manifest.get("dataset", {})
    if not dataset.get("config_hash_matches_run"):
        raise ExportError("Run dataset identity is stale or not validated")


def validate_exported_run_artifacts(export_cfg: dict[str, Any], run_dir: Path, variables: dict[str, str], manifest: dict[str, Any]) -> None:
    """Bind every allowlisted run file to the SHA-256 recorded by the runner."""
    recorded = {
        Path(artifact["path"]).resolve(): artifact.get("sha256")
        for step in manifest.get("steps", [])
        for artifact in step.get("artifacts", [])
        if artifact.get("sha256")
    }
    for item in export_cfg["include"]:
        raw = expand(str(item["from"]), variables)
        source = Path(raw)
        if not source.is_absolute():
            continue
        source = source.resolve()
        if is_within(source, run_dir) and recorded.get(source) != sha256(source):
            raise ExportError(f"Run artifact is absent from the manifest or has a hash mismatch: {source}")


def expand(value: str, variables: dict[str, str]) -> str:
    """Replace ${NAME} variables in an export path or command string."""
    for key, replacement in variables.items():
        value = value.replace("${" + key + "}", replacement)
    return value


def is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` equals ``root`` or lies below it."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_name(value: str, label: str) -> str:
    """Require a single safe path segment for package names and release IDs."""
    if not NAME_PATTERN.fullmatch(value) or value == LATEST_POINTER:
        raise ExportError(
            f"Invalid {label} {value!r}: use 1-128 letters, digits, '.', '_' or '-', "
            f"starting with a letter or digit, and not {LATEST_POINTER!r}."
        )
    return value


def default_release_id() -> str:
    """Build a unique, time-sortable release ID."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def protected_roots(cfg: dict[str, Any], project_root: Path, run_dir: Path | None) -> dict[str, Path]:
    """Collect run and data directories that an export must never overlap."""
    roots = {"runs root": (project_root / str(cfg.get("outputs", {}).get("runs_root", "runs"))).resolve()}
    data_dir = cfg.get("data", {}).get("output_dir")
    if data_dir:
        roots["data directory"] = (project_root / str(data_dir)).resolve()
    if run_dir is not None:
        roots["run directory"] = run_dir
    return roots


def resolve_package_root(
    output: Path | None,
    project_root: Path,
    package_name: str,
    protected: dict[str, Path],
) -> Path:
    """Resolve the package directory and refuse anything outside paper/releases/."""
    releases_root = (project_root / "paper" / "releases").resolve()
    if is_within(project_root, releases_root):
        raise ExportError(f"Refusing to export: releases root {releases_root} resolves to the project root or an ancestor.")

    target = output if output is not None else releases_root / package_name
    resolved = target.resolve()
    if is_within(project_root, resolved):
        raise ExportError(f"Refusing --output {resolved}: it is the project root or one of its ancestors.")
    for label, root in protected.items():
        if is_within(resolved, root) or is_within(root, resolved):
            raise ExportError(f"Refusing --output {resolved}: it overlaps the {label} {root}.")
    if resolved == releases_root or not is_within(resolved, releases_root):
        raise ExportError(
            f"Refusing --output {target} (resolves to {resolved}): exports must be a package directory under {releases_root}."
        )
    if resolved.exists() and not resolved.is_dir():
        raise ExportError(f"Refusing --output {resolved}: it exists and is not a directory.")
    return resolved


def ensure_under_root(path: Path, root: Path, allow_run_dir: Path | None) -> Path:
    """Reject export sources outside the project root or selected run dir."""
    resolved = path.resolve()
    roots = [root.resolve()]
    if allow_run_dir is not None:
        roots.append(allow_run_dir.resolve())

    for allowed in roots:
        if is_within(resolved, allowed):
            return resolved
    raise ExportError(f"Refusing to export path outside allowed roots: {path} -> {resolved}")


def safe_dest(root: Path, relative: str) -> Path:
    """Resolve an artifact destination while preventing path traversal."""
    dest = (root / relative).resolve()
    if dest == root.resolve() or not is_within(dest, root.resolve()):
        raise ExportError(f"Unsafe export destination: {relative}")
    return dest


def copy_allowlisted(
    item: dict[str, Any],
    project_root: Path,
    run_dir: Path | None,
    release_root: Path,
    variables: dict[str, str],
) -> list[Path]:
    """Copy one explicitly allowlisted file or directory into the release."""
    src_raw = expand(str(item["from"]), variables)
    dst_raw = expand(str(item["to"]), variables)

    src = Path(src_raw)
    if not src.is_absolute():
        src = project_root / src
    src = ensure_under_root(src, project_root, run_dir)

    dst = safe_dest(release_root, dst_raw)

    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        raise ExportError(f"Duplicate export destination: {dst_raw}")

    written: list[Path] = []
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written.append(dst)
    elif src.is_dir():
        # Directory export is still explicit: only this listed directory is copied.
        # Use sparingly; for strict minimal packages prefer file-level mappings.
        # copytree follows symlinks, so every entry must stay inside the allowed roots.
        for path in src.rglob("*"):
            ensure_under_root(path, project_root, run_dir)
        shutil.copytree(src, dst)
        written.extend([p for p in dst.rglob("*") if p.is_file()])
    else:
        raise ExportError(f"Unsupported source type: {src}")
    return written


def sha256(path: Path) -> str:
    """Compute a streaming SHA-256 hash for one exported file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_public_config(cfg: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Keep only public config sections and drop private execution metadata."""
    public: dict[str, Any] = {}
    for key in keys:
        if key in cfg:
            public[key] = cfg[key]
    # paper_export may contain private internal source paths; do not expose it.
    public.pop("paper_export", None)
    public.pop("flow", None)
    return public


def write_new(path: Path, text: str) -> Path:
    """Write a generated file, refusing to replace an allowlisted file of the same name."""
    try:
        with path.open("x", encoding="utf-8") as f:
            f.write(text)
    except FileExistsError as exc:
        raise ExportError(f"Export destination collides with generated file: {path.name}") from exc
    return path


def payload_files(payload_root: Path) -> set[str]:
    """List payload files relative to the payload root, rejecting symlinks."""
    files: set[str] = set()
    for path in payload_root.rglob("*"):
        if path.is_symlink():
            raise ExportError(f"Release payload contains a symlink: {path}")
        if path.is_file():
            files.add(path.relative_to(payload_root).as_posix())
    return files


def build_payload(
    payload_root: Path,
    cfg: dict[str, Any],
    project_root: Path,
    run_dir: Path | None,
    variables: dict[str, str],
    package_name: str,
    release_id: str,
) -> None:
    """Copy allowlisted files and write public config, README, metadata and hash manifest."""
    export_cfg = cfg.get("paper_export", {})
    for item in export_cfg.get("include", []):
        if not isinstance(item, dict) or "from" not in item or "to" not in item:
            raise ExportError("Every paper_export.include item requires 'from' and 'to'.")
        copy_allowlisted(
            item=item,
            project_root=project_root,
            run_dir=run_dir,
            release_root=payload_root,
            variables=variables,
        )

    # Export only selected top-level config sections.
    public_keys = export_cfg.get(
        "public_config_keys",
        ["experiment", "topic", "qec", "noise", "data", "model", "training", "objective", "adaptation", "benchmarks"],
    )
    public_cfg = select_public_config(cfg, list(public_keys))
    write_new(payload_root / "config.yaml", yaml.safe_dump(public_cfg, sort_keys=False, allow_unicode=True))

    # README generated from public metadata only.
    reproduce_cmd = export_cfg.get("reproduce_command", "python reproduce.py --config config.yaml")
    readme = f"""# {package_name}

Minimal reproducible artifact generated from the private AI-QEC research repository.

## Scope

{export_cfg.get("scope", "Evaluation / benchmark reproduction")}

## Reproduce

```bash
{reproduce_cmd}
```

This package intentionally contains only allowlisted files required for reproduction.
The original repository structure is not exported.
"""
    write_new(payload_root / "README.md", readme)

    metadata = {
        "package_name": package_name,
        "release_id": release_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": variables["RUN_ID"],
        "scope": export_cfg.get("scope"),
    }
    write_new(payload_root / METADATA_NAME, json.dumps(metadata, indent=2, ensure_ascii=False))

    # Hash manifest covers every payload file except itself.
    manifest = {rel: sha256(payload_root / rel) for rel in sorted(payload_files(payload_root))}
    write_new(payload_root / MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))


def verify_payload(payload_root: Path) -> dict[str, str]:
    """Check that the hash manifest lists exactly the payload files with matching hashes."""
    manifest = json.loads((payload_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    on_disk = payload_files(payload_root) - {MANIFEST_NAME}
    if set(manifest) != on_disk:
        raise ExportError(f"Manifest does not match payload files: {sorted(set(manifest) ^ on_disk)}")
    for rel, digest in manifest.items():
        if sha256(payload_root / rel) != digest:
            raise ExportError(f"Hash mismatch after build: {rel}")
    return manifest


def write_zip(payload_root: Path, zip_path: Path, package_name: str) -> None:
    """Archive the payload under a top-level folder named after the package."""
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(payload_files(payload_root)):
            zf.write(payload_root / rel, arcname=f"{package_name}/{rel}")


def verify_zip(zip_path: Path, payload_root: Path, package_name: str, manifest: dict[str, str]) -> None:
    """Check that the archive holds exactly the payload files with manifest hashes."""
    expected = {f"{package_name}/{rel}": rel for rel in payload_files(payload_root)}
    with zipfile.ZipFile(zip_path) as zf:
        if set(zf.namelist()) != set(expected):
            raise ExportError("Zip archive does not match payload files.")
        if zf.testzip() is not None:
            raise ExportError("Zip archive failed CRC check.")
        manifest_bytes = (payload_root / MANIFEST_NAME).read_bytes()
        for name, rel in expected.items():
            h = hashlib.sha256()
            with zf.open(name) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            digest = hashlib.sha256(manifest_bytes).hexdigest() if rel == MANIFEST_NAME else manifest[rel]
            if h.hexdigest() != digest:
                raise ExportError(f"Zip member hash mismatch: {name}")


def write_release_index(release_dir: Path, release_id: str, payload_root: Path, zip_enabled: bool) -> None:
    """Write hashes outside the payload, avoiding a self-referential manifest."""
    index = {
        "release_id": release_id,
        "payload_manifest": f"{payload_root.name}/{MANIFEST_NAME}",
        "payload_manifest_sha256": sha256(payload_root / MANIFEST_NAME),
        "zip_sha256": sha256(release_dir / f"{payload_root.name}.zip") if zip_enabled else None,
    }
    (release_dir / RELEASE_INDEX_NAME).write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def verify_release(release_dir: Path) -> None:
    """Verify a committed release and its external hash index."""
    index_path = release_dir / RELEASE_INDEX_NAME
    if not index_path.is_file():
        raise ExportError(f"Release index missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    payload_root = release_dir / str(index["payload_manifest"]).split("/", 1)[0]
    manifest = verify_payload(payload_root)
    if sha256(payload_root / MANIFEST_NAME) != index.get("payload_manifest_sha256"):
        raise ExportError("Release index manifest hash mismatch")
    zip_path = release_dir / f"{payload_root.name}.zip"
    if index.get("zip_sha256") is not None:
        if not zip_path.is_file() or sha256(zip_path) != index["zip_sha256"]:
            raise ExportError("Release index zip hash mismatch")
        verify_zip(zip_path, payload_root, payload_root.name, manifest)


def fsync_dir(path: Path) -> None:
    """Persist a directory entry change where the platform supports it."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def update_latest(package_root: Path, release_id: str) -> None:
    """Atomically point LATEST at a committed release."""
    tmp = package_root / f".{LATEST_POINTER}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp.open("x", encoding="utf-8") as f:
            f.write(release_id + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, package_root / LATEST_POINTER)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_dir(package_root)


def publish_release(package_root: Path, release_id: str, build: Callable[[Path], None]) -> Path:
    """Build a release in sibling staging, commit it by rename, then move LATEST.

    Existing releases and the LATEST pointer are left untouched unless the new
    release is fully built, verified and committed.
    """
    release_dir = package_root / release_id
    if release_dir.exists() or release_dir.is_symlink():
        raise ExportError(f"Release {release_id!r} already exists at {release_dir}; releases are immutable.")

    package_root.mkdir(parents=True, exist_ok=True)
    staging = package_root / f"{STAGING_PREFIX}{release_id}-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        build(staging)
        if release_dir.exists() or release_dir.is_symlink():
            raise ExportError(f"Release {release_id!r} appeared during the build; refusing to replace it.")
        os.rename(staging, release_dir)
    except BaseException:
        # Only the staging directory created above is removed; committed releases are never touched.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    fsync_dir(package_root)
    update_latest(package_root, release_id)
    return release_dir


def main() -> int:
    """Build a verified, immutable paper artifact release and point LATEST at it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Package directory under <project>/paper/releases/ (default: paper/releases/<package_name>)",
    )
    parser.add_argument("--release-id", default=None, help="New release ID (default: UTC timestamp + random suffix)")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--verify", type=Path, default=None, help="Verify an existing immutable release and exit")
    args = parser.parse_args()

    if args.verify is not None:
        try:
            verify_release(args.verify.resolve())
        except (ExportError, OSError, ValueError, KeyError) as exc:
            print(f"Release verification failed: {exc}", file=sys.stderr)
            return EXPORT_ERROR_EXIT_CODE
        print(f"Release verified: {args.verify.resolve()}")
        return 0

    if args.config is None:
        parser.error("--config is required unless --verify is used")

    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir else None
    try:
        cfg = load_config(config_path, project_root=project_root)
    except (OSError, ConfigValidationError) as exc:
        print(f"Export failed: invalid config: {exc}", file=sys.stderr)
        return EXPORT_ERROR_EXIT_CODE

    export_cfg = cfg.get("paper_export")
    if export_cfg is None:
        print("Export failed: config has no paper_export closure declaration.", file=sys.stderr)
        return EXPORT_ERROR_EXIT_CODE
    if not export_cfg["enabled"]:
        raise SystemExit("paper_export.enabled is false.")

    try:
        run_id = run_dir.name if run_dir else "no_run"
        package_name = validate_name(
            str(export_cfg.get("package_name", cfg.get("experiment", {}).get("name", "paper_artifact"))),
            "package_name",
        )
        release_id = validate_name(args.release_id or default_release_id(), "release ID")
        package_root = resolve_package_root(
            args.output, project_root, package_name, protected_roots(cfg, project_root, run_dir)
        )
        validate_run_closure(cfg, run_dir)

        variables = {
            "PROJECT_ROOT": str(project_root),
            "RUN_DIR": str(run_dir) if run_dir else "",
            "RUN_ID": run_id,
            "CONFIG": str(config_path),
        }
        assert run_dir is not None
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        validate_exported_run_artifacts(export_cfg, run_dir, variables, run_manifest)

        def build(staging: Path) -> None:
            payload_root = staging / package_name
            payload_root.mkdir()
            build_payload(payload_root, cfg, project_root, run_dir, variables, package_name, release_id)
            manifest = verify_payload(payload_root)
            if not args.no_zip:
                zip_path = staging / f"{package_name}.zip"
                write_zip(payload_root, zip_path, package_name)
                verify_zip(zip_path, payload_root, package_name, manifest)
            write_release_index(staging, release_id, payload_root, not args.no_zip)
            verify_release(staging)

        release_dir = publish_release(package_root, release_id, build)
        verify_release(release_dir)
    except (ExportError, OSError, ValueError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return EXPORT_ERROR_EXIT_CODE

    print(f"Paper artifact: {release_dir / package_name}")
    if not args.no_zip:
        print(f"ZIP: {release_dir / (package_name + '.zip')}")
    print(f"{LATEST_POINTER} -> {release_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
