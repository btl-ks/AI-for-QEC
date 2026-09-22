"""Canonical JSON and SHA-256 digests used for identities and integrity checks."""

from collections.abc import Mapping, Sequence
import dataclasses
from enum import Enum
import hashlib
import json
from pathlib import Path


def to_jsonable(value: object) -> object:
    """Convert dataclasses, mappings, sequences, and enums into plain JSON values."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(child) for key, child in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [to_jsonable(child) for child in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON-serializable: {value!r}")


def canonical_json(value: object) -> str:
    """Serialize with sorted keys and no insignificant whitespace."""

    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: object) -> str:
    """Return ``sha256:<hex>`` of the canonical JSON form."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return ``sha256:<hex>`` of a file's bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def derive_seed(master_seed: int, name: str) -> int:
    """Derive a stable 63-bit seed for a named stream."""

    material = f"{int(master_seed)}:{name}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") >> 1


def write_json_atomic(path: Path, value: object) -> None:
    """Write indented JSON through a temporary file and ``os.replace``."""

    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))
