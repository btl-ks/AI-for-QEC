"""Run provenance: git working-tree state, package versions, dataset references."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from ai_qec.utils.config import config_hash, data_output_dir
from ai_qec.utils.serialization import read_json

KEY_PACKAGES = ("numpy", "PyYAML", "stim", "pymatching", "torch")
MAX_RECORDED_DIRTY_PATHS = 200
MAX_UNTRACKED_PATCH_BYTES = 10 * 1024 * 1024


def _git(project_root: Path, *args: str) -> bytes | None:
    """Run one git command in project_root and return stdout, or None on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout


def _text(value: bytes | None) -> str | None:
    """Decode one-line git output."""
    # Choose the first expression when value is None; otherwise use the fallback.
    return None if value is None else value.decode("utf-8", "replace").strip()


def git_diff(project_root: str | Path) -> tuple[bytes, list[str]] | None:
    """Return a patch of all uncommitted changes against HEAD, or None if unavailable.

    The patch holds tracked-file changes plus untracked, non-ignored files as new
    files, so ``git apply`` on a checkout of HEAD restores the working tree.
    Untracked files larger than MAX_UNTRACKED_PATCH_BYTES are left out and
    returned as the second element.
    """
    top = _text(_git(Path(project_root), "rev-parse", "--show-toplevel"))
    # Return early when top is None.
    if top is None:
        return None
    root = Path(top)
    tracked = _git(root, "diff", "--binary", "HEAD")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    # Return early when tracked is None or untracked is None.
    if tracked is None or untracked is None:
        return None

    parts = [tracked]
    skipped: list[str] = []
    # Keep only values that satisfy raw.
    for name in sorted(os.fsdecode(raw) for raw in untracked.split(b"\0") if raw):
        # Follow this branch when (root / name).lstat().st_size > MAX_UNTRACKED_PATCH_BYTES.
        if (root / name).lstat().st_size > MAX_UNTRACKED_PATCH_BYTES:
            skipped.append(name)
            continue
        # --no-index exits 1 when the files differ, which is always the case here.
        proc = subprocess.run(
            ["git", "diff", "--binary", "--no-index", "--", "/dev/null", name],
            cwd=root,
            capture_output=True,
        )
        # Return early when proc.returncode not in (0, 1).
        if proc.returncode not in (0, 1):
            return None
        parts.append(proc.stdout)
    return b"".join(parts), skipped


def git_state(project_root: str | Path) -> dict[str, Any]:
    """Describe the git working tree containing project_root.

    ``dirty`` is None when git status cannot be read. ``diff_sha256`` hashes the
    patch returned by ``git_diff``, which covers tracked changes and untracked,
    non-ignored files except those listed in ``patch_skipped_paths``.
    """
    root = Path(project_root)
    inside = _git(root, "rev-parse", "--is-inside-work-tree")
    # Return early when inside is None or inside.strip() != b'true'.
    if inside is None or inside.strip() != b"true":
        return {"available": False}

    commit = _git(root, "rev-parse", "--verify", "HEAD")
    # Choose the first expression when commit; otherwise use the fallback.
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") if commit else None
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    # Choose the first expression when status is None; otherwise use the fallback.
    # Keep only values that satisfy line.
    dirty_paths = None if status is None else [
        line for line in status.decode("utf-8", "replace").splitlines() if line
    ]
    # Choose the first expression when commit; otherwise use the fallback.
    patch = git_diff(root) if commit else None

    # Choose the first expression when dirty_paths is None; otherwise use the fallback.
    # Choose the first expression when patch and patch[0]; otherwise use the fallback.
    # Choose the first expression when patch; otherwise use the fallback.
    return {
        "available": True,
        "commit": _text(commit),
        "branch": _text(branch),
        "dirty": None if dirty_paths is None else bool(dirty_paths),
        "dirty_paths": None if dirty_paths is None else dirty_paths[:MAX_RECORDED_DIRTY_PATHS],
        "dirty_paths_truncated": len(dirty_paths or []) > MAX_RECORDED_DIRTY_PATHS,
        "diff_sha256": hashlib.sha256(patch[0]).hexdigest() if patch and patch[0] else None,
        "patch_skipped_paths": patch[1] if patch else [],
    }


def clean_worktree_error(git: dict[str, Any], max_listed: int = 10) -> str | None:
    """Explain why a working tree cannot anchor a formal run, or return None if clean."""
    # Return early when not git.get('available').
    if not git.get("available"):
        return "project root is not inside a git working tree"
    # Return early when git.get('commit') is None.
    if git.get("commit") is None:
        return "git repository has no commits"
    # Return early when git.get('dirty') is None.
    if git.get("dirty") is None:
        return "git status could not be read"
    # Follow this branch when git['dirty'].
    if git["dirty"]:
        paths = [path.strip() for path in git.get("dirty_paths") or []]
        listed = "; ".join(paths[:max_listed])
        # Choose the first expression when len(paths) > max_listed; otherwise use the fallback.
        more = f"; ... (+{len(paths) - max_listed} more)" if len(paths) > max_listed else ""
        return f"working tree has uncommitted changes: {listed}{more}"
    return None


def package_versions(names: tuple[str, ...] = KEY_PACKAGES) -> dict[str, str | None]:
    """Return installed versions of the named distributions; None when absent."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def code_version(project_root: str | Path) -> str | None:
    """Return the committed source identity used to bind checkpoints to code."""
    return _text(_git(Path(project_root), "rev-parse", "HEAD"))


def installed_distributions() -> dict[str, str]:
    """Return every installed distribution name and version, sorted by name."""
    found: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        # Follow this branch when name.
        if name:
            found[name] = dist.version
    return dict(sorted(found.items(), key=lambda item: item[0].lower()))


def interpreter_environment(
    executable: str | Path,
    names: tuple[str, ...] = KEY_PACKAGES,
) -> dict[str, Any]:
    """Record Python and installed distributions from an explicit interpreter.

    The experiment runner may execute flow steps through a Conda environment
    different from its own interpreter.  Querying the target interpreter keeps
    the manifest tied to the packages that actually executed the experiment.
    """
    probe = "\n".join(
        (
            "import importlib.metadata as metadata",
            "import json",
            "import platform",
            "import sys",
            f"names = {list(names)!r}",
            "packages = {}",
            "for name in names:",
            "    try:",
            "        packages[name] = metadata.version(name)",
            "    except metadata.PackageNotFoundError:",
            "        packages[name] = None",
            "distributions = {}",
            "for dist in metadata.distributions():",
            "    name = dist.metadata['Name']",
            "    if name:",
            "        distributions[name] = dist.version",
            "print(json.dumps({",
            "    'python': sys.version,",
            "    'executable': sys.executable,",
            "    'prefix': sys.prefix,",
            "    'platform': platform.platform(),",
            "    'packages': packages,",
            "    'distributions': dict(sorted(distributions.items(), key=lambda item: item[0].lower())),",
            "}))",
        )
    )
    try:
        proc = subprocess.run(
            [str(executable), "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        # Choose the first expression when isinstance(exc, subprocess.CalledProcessError) and exc.stderr; otherwise use the fallback.
        stderr = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else ""
        # Choose the first expression when stderr; otherwise use the fallback.
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"could not inspect interpreter {executable}{detail}") from exc
    try:
        snapshot = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"interpreter {executable} did not return a valid environment snapshot") from exc
    required = {"python", "executable", "prefix", "platform", "packages", "distributions"}
    # Reject this state when not isinstance(snapshot, dict) or set(snapshot) != required.
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise RuntimeError(f"interpreter {executable} returned an incomplete environment snapshot")
    # Reject this state when the invalid compound condition is detected.
    if not isinstance(snapshot["packages"], dict) or not isinstance(snapshot["distributions"], dict):
        raise RuntimeError(f"interpreter {executable} returned malformed package metadata")
    return snapshot


def dataset_reference(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Link a run to the dataset manifest its config points at.

    ``config_hash_matches_run`` is False when the dataset on disk was generated
    from a different config, e.g. a stale dataset reused via ``--only train``.
    """
    dataset_dir = data_output_dir(config, project_root)
    manifest_path = dataset_dir / "dataset_manifest.json"
    reference: dict[str, Any] = {"dir": str(dataset_dir), "manifest": None}
    # Return early when not manifest_path.exists().
    if not manifest_path.exists():
        return reference

    manifest = read_json(manifest_path)
    dataset_hash = manifest.get("config_hash")
    reference.update(
        {
            "manifest": str(manifest_path),
            "dataset_id": manifest.get("dataset_id"),
            "created_at": manifest.get("context", {}).get("created_at"),
            "config_hash": dataset_hash,
            "config_hash_matches_run": dataset_hash == config_hash(config),
        }
    )
    return reference
