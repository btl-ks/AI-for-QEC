#!/usr/bin/env python3
"""Generate AI-QEC datasets from an experiment config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _bootstrap(project_root: Path) -> None:
    """Make the local package importable without installation."""
    sys.path.insert(0, str(project_root))


def main() -> int:
    """Parse CLI arguments and generate the configured dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    _bootstrap(project_root)

    from ai_qec.data.generators.qec_generator import generate_dataset
    from ai_qec.utils.config import load_config

    config = load_config(args.config, project_root=project_root)
    manifest = generate_dataset(
        config=config,
        project_root=project_root,
    )
    from ai_qec.utils.config import data_output_dir
    print(f"Dataset: {data_output_dir(config, project_root)}")
    print(f"Samples: {manifest['sample_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
