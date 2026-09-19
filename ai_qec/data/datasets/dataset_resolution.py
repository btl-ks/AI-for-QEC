"""Decide which dataset a run uses, and record that decision on the run.

A run reaches its data in one of three ways: it references a source run's dataset,
it reuses an immutable dataset already on disk, or it generates one. All three end
at the same verifier and leave the same trace in the run manifest. New data uses
the same split sampler as the script entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ai_qec.data.datasets.toric_dataset import (
    build_toric_dataset_manifest,
    staged_dataset_dir,
    validate_toric_dataset,
)
from ai_qec.data.generators.toric_generator import sample_toric_splits
from ai_qec.utils.config import data_output_dir, write_json
from ai_qec.utils.source_run import SourceRun, link_source_dataset, validate_source_dataset


class _Run(Protocol):
    """The part of a run record this module needs."""

    def refresh_dataset(self) -> None: ...


def resolve_dataset(
    run: _Run,
    config: dict[str, Any],
    project_root: str | Path,
    *,
    step: dict[str, Any],
    source_run: SourceRun | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Return the validated dataset directory and manifest for this run.

    New data uses the same sampler as the script entry point.  The staged directory
    is committed atomically and verified, so failed generation leaves no dataset.
    """
    dataset_dir = data_output_dir(config, project_root)
    # Follow this branch when source_run is not None.
    if source_run is not None:
        dataset_dir, manifest = validate_source_dataset(source_run, config)
        step["reused_immutable"] = True
        step["source_run"] = source_run.run_id
    # Handle all remaining cases.
    else:
        # Follow this branch when dataset_dir.exists().
        if dataset_dir.exists():
            step["reused_immutable"] = True
        # Handle all remaining cases.
        else:
            with staged_dataset_dir(dataset_dir) as staging:
                files, code, circuit = sample_toric_splits(config, staging)
                # Reject this state when not files.
                if not files:
                    raise ValueError("sampler produced no dataset splits")
                write_json(
                    staging / "dataset_manifest.json",
                    build_toric_dataset_manifest(
                        config, code, files,
                        extra_context={"circuit": circuit.context()},
                    ),
                )
        manifest = validate_toric_dataset(dataset_dir, config)
    # Link the run to its dataset before registering the artifact: both helpers write
    # the manifest, and a pending ``outputs`` entry is still a Path at that point.
    if source_run is None:
        run.refresh_dataset()
    # Handle all remaining cases.
    else:
        link_source_dataset(run, source_run)
    step["outputs"].append(dataset_dir / "dataset_manifest.json")
    return dataset_dir, manifest
