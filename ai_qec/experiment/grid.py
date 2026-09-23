"""Deterministic expansion of one base configuration into a grid of experiments.

Overrides address existing leaf fields by dotted path (``"qec.distance"``), so a
misspelled field fails instead of silently adding a key nobody reads. Every
field has exactly one source: shared overrides, an axis, or an axis-coupled
override; ``experiment.name`` comes from the name template.
"""

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import itertools
from typing import Any

from ai_qec.registry.validation import ConfigurationError, MissingConfigurationError

NAME_PATH = "experiment.name"


@dataclass(frozen=True, slots=True)
class GridPoint:
    """Axis values of one grid point (keyed by axis alias) and its full configuration."""

    values: Mapping[str, Any]
    config: dict[str, object]


def _parent(config: Mapping[str, object], path: str) -> tuple[dict, str]:
    parts = path.split(".")
    node = config
    for index, part in enumerate(parts[:-1]):
        child = node.get(part) if isinstance(node, Mapping) else None
        if not isinstance(child, Mapping):
            raise MissingConfigurationError((".".join(parts[: index + 1]),))
        node = child
    if not isinstance(node, Mapping) or parts[-1] not in node:
        raise MissingConfigurationError((path,))
    if isinstance(node[parts[-1]], Mapping):
        raise ConfigurationError(f"[{path}] is a section; override its individual fields")
    return node, parts[-1]


def with_overrides(
    config: Mapping[str, object], overrides: Mapping[str, object]
) -> dict[str, object]:
    """Return a deep copy of ``config`` with each existing dotted-path field replaced."""

    result = copy.deepcopy(dict(config))
    for path, value in overrides.items():
        parent, key = _parent(result, path)
        parent[key] = copy.deepcopy(value)
    return result


def config_grid(
    base: Mapping[str, object],
    *,
    name: str,
    axes: Mapping[str, tuple[str, Sequence[object]]],
    coupled: Mapping[str, Mapping[object, Mapping[str, object]]] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> tuple[GridPoint, ...]:
    """Cartesian product of ``axes`` (first axis outermost) applied to ``base``.

    ``axes`` maps an alias to ``(dotted path, values)``; ``coupled`` maps an alias to
    per-value overrides; ``overrides`` apply to every point; ``name`` is a
    ``str.format`` template over the aliases and becomes ``experiment.name``.
    """

    coupled = dict(coupled or {})
    overrides = dict(overrides or {})
    if not axes:
        raise ConfigurationError("config_grid requires at least one axis")
    for alias in axes:
        if not alias.isidentifier():
            raise ConfigurationError(f"axis alias {alias!r} must be a Python identifier")
    unknown = sorted(set(coupled) - set(axes))
    if unknown:
        raise ConfigurationError(f"coupled overrides refer to unknown axes: {unknown}")

    sources: dict[str, str] = {NAME_PATH: "name template"}
    for path in overrides:
        sources.setdefault(path, "overrides")
        if sources[path] != "overrides":
            raise ConfigurationError(f"[{path}] is set by both overrides and the {sources[path]}")
    for alias, (path, values) in axes.items():
        if path in sources:
            raise ConfigurationError(f"[{path}] is set by both axis {alias!r} and {sources[path]}")
        sources[path] = f"axis {alias!r}"
        if not values:
            raise ConfigurationError(f"axis {alias!r} has no values")
    for alias, by_value in coupled.items():
        missing = [value for value in axes[alias][1] if value not in by_value]
        if missing:
            raise ConfigurationError(f"coupled overrides for axis {alias!r} lack values {missing}")
        for value, fields in by_value.items():
            for path in fields:
                owner = sources.get(path)
                if owner is not None and owner != f"coupled {alias!r}":
                    raise ConfigurationError(
                        f"[{path}] is set by both coupled {alias!r} and {owner}"
                    )
                sources[path] = f"coupled {alias!r}"

    shared = with_overrides(base, overrides)
    points: list[GridPoint] = []
    aliases = list(axes)
    for combination in itertools.product(*(axes[alias][1] for alias in aliases)):
        values = dict(zip(aliases, combination))
        fields = {axes[alias][0]: value for alias, value in values.items()}
        for alias, by_value in coupled.items():
            fields.update(by_value[values[alias]])
        fields[NAME_PATH] = name.format(**values)
        points.append(GridPoint(values=values, config=with_overrides(shared, fields)))

    names = [point.config["experiment"]["name"] for point in points]
    duplicates = sorted({item for item in names if names.count(item) > 1})
    if duplicates:
        raise ConfigurationError(
            f"name template {name!r} produces duplicate experiment names: {duplicates}"
        )
    return tuple(points)
