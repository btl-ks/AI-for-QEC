"""Decide which dataset a run uses, and record that decision on the run.

A run reaches its data in one of three ways: it references a source run's dataset,
it reuses an immutable dataset already on disk, or it generates one.  All three end
at the same verifier, and all three have to leave the same trace in the run manifest.
Entry points keep the sampling loop -- the part that mirrors the paper -- and hand it
here as ``sample_splits`` so the branching does not have to be written out again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from ai_qec.data.datasets.toric_dataset import (
    build_toric_dataset_manifest,
    staged_dataset_dir,
    validate_toric_dataset,
)
from ai_qec.qec.circuits.registry import build_circuit
from ai_qec.qec.codes.registry import build_code
from ai_qec.utils.config import data_output_dir, write_json
from ai_qec.utils.source_run import SourceRun, link_source_dataset, validate_source_dataset


SampleSplits = Callable[[Path], dict[str, dict[str, Any]]]


class _Run(Protocol):
    """The part of a run record this module needs."""

    def refresh_dataset(self) -> None: ...


def resolve_dataset(
    run: _Run,
    config: dict[str, Any],
    project_root: str | Path,
    *,
    step: dict[str, Any],
    sample_splits: SampleSplits,
    source_run: SourceRun | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Return the validated dataset directory and manifest for this run.

    ``sample_splits`` is called only when a new dataset has to be generated.  It
    receives the staging directory and returns the manifest file records, one per
    split, exactly as :func:`write_toric_split` produces them.  Whatever it writes is
    committed atomically and then verified, so a failed generation leaves nothing.
    """
    dataset_dir = data_output_dir(config, project_root)
    if source_run is not None:
        dataset_dir, manifest = validate_source_dataset(source_run, config)
        step["reused_immutable"] = True
        step["source_run"] = source_run.run_id
    else:
        if dataset_dir.exists():
            step["reused_immutable"] = True
        else:
            code = build_code(config)
            with staged_dataset_dir(dataset_dir) as staging:
                files = sample_splits(staging)
                if not files:
                    raise ValueError("sample_splits produced no dataset splits")
                write_json(
                    staging / "dataset_manifest.json",
                    build_toric_dataset_manifest(
                        config, code, files,
                        extra_context={"circuit": build_circuit(config, code).context()},
                    ),
                )
        manifest = validate_toric_dataset(dataset_dir, config)
    # Link the run to its dataset before registering the artifact: both helpers write
    # the manifest, and a pending ``outputs`` entry is still a Path at that point.
    if source_run is None:
        run.refresh_dataset()
    else:
        link_source_dataset(run, source_run)
    step["outputs"].append(dataset_dir / "dataset_manifest.json")
    return dataset_dir, manifest
