"""Fail-fast configuration checks used before any constructor has side effects."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Final

from ai_qec.registries import REGISTRIES_BY_PATH
from ai_qec.registry import Registry, UnknownRegistrationError


UNRESOLVED: Final = "unresolved"
_MISSING: Final = object()


class ConfigurationError(ValueError):
    """Base error for invalid pre-construction configuration."""


class UnresolvedConfigurationError(ConfigurationError):
    def __init__(self, paths: Iterable[str]) -> None:
        self.paths = tuple(sorted(paths))
        joined = "\n  - ".join(self.paths)
        super().__init__(f"unresolved configuration values:\n  - {joined}")


class MissingConfigurationError(ConfigurationError):
    def __init__(self, paths: Iterable[str]) -> None:
        self.paths = tuple(sorted(paths))
        joined = "\n  - ".join(self.paths)
        super().__init__(f"missing required configuration values:\n  - {joined}")


def find_unresolved(value: object, path: str = "") -> Iterator[str]:
    """Yield complete paths whose exact string value is ``unresolved``."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from find_unresolved(child, child_path)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            yield from find_unresolved(child, child_path)
        return

    if value == UNRESOLVED:
        yield path


def validate_no_unresolved(
    config: Mapping[str, object],
    *,
    allow: Iterable[str] = (),
) -> None:
    """Reject every unresolved path not explicitly allowed by the caller."""

    allowed = frozenset(allow)
    missing = tuple(path for path in find_unresolved(config) if path not in allowed)
    if missing:
        raise UnresolvedConfigurationError(missing)


def get_config_path(config: Mapping[str, object], path: str) -> object:
    """Read a dotted mapping path and raise a precise error when it is absent."""

    current: object = config
    traversed: list[str] = []
    for part in path.split("."):
        traversed.append(part)
        if not isinstance(current, Mapping) or part not in current:
            raise MissingConfigurationError((".".join(traversed),))
        current = current[part]
    return current


def _optional_config_path(config: Mapping[str, object], path: str) -> object:
    current: object = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def validate_registered_selections(
    config: Mapping[str, object],
    *,
    registries: Mapping[str, Registry[object]] = REGISTRIES_BY_PATH,
    required: Iterable[str] = (),
) -> None:
    """Check present selections without calling any registered factory."""

    required_paths = frozenset(required)
    missing = [path for path in required_paths if _optional_config_path(config, path) is _MISSING]
    if missing:
        raise MissingConfigurationError(missing)

    for path, registry in registries.items():
        selected = _optional_config_path(config, path)
        if selected is _MISSING:
            continue
        values: tuple[object, ...]
        if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes, bytearray)):
            values = tuple(selected)
        else:
            values = (selected,)
        for value in values:
            if not isinstance(value, str):
                raise ConfigurationError(f"[{path}] selection must be a string, got {value!r}")
            try:
                registry.require(value)
            except UnknownRegistrationError as error:
                raise ConfigurationError(str(error)) from error


def validate_config(
    config: Mapping[str, object],
    *,
    allow_unresolved: Iterable[str] = (),
    required_selections: Iterable[str] = (),
    registries: Mapping[str, Registry[object]] = REGISTRIES_BY_PATH,
) -> None:
    """Run unresolved and registered-selection checks before construction."""

    validate_no_unresolved(config, allow=allow_unresolved)
    validate_registered_selections(
        config,
        registries=registries,
        required=required_selections,
    )


def build_from_config(
    config: Mapping[str, object],
    path: str,
    *,
    registries: Mapping[str, Registry[object]] = REGISTRIES_BY_PATH,
    **kwargs: object,
) -> object:
    """Build one scalar selection through its declared Registry."""

    try:
        registry = registries[path]
    except KeyError as error:
        raise ConfigurationError(f"no Registry declared for configuration path {path!r}") from error
    selected = get_config_path(config, path)
    if selected == UNRESOLVED:
        raise UnresolvedConfigurationError((path,))
    if not isinstance(selected, str):
        raise ConfigurationError(f"[{path}] selection must be a string, got {selected!r}")
    return registry.build(selected, **kwargs)
