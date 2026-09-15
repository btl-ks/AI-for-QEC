"""Small wrappers around project serialization formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return data


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a UTF-8 JSON object with stable indentation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
