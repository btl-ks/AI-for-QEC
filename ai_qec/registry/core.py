"""Small, dependency-free registry used by configuration-driven adapters."""

from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Generic, TypeVar

T = TypeVar("T")
Factory = Callable[..., T]


class RegistryError(ValueError):
    """Base error for registry declaration and lookup failures."""


class DuplicateRegistrationError(RegistryError):
    """Raised before an existing registration could be overwritten."""


class UnknownRegistrationError(RegistryError):
    """Raised when configuration selects a key with no executable factory."""


class Registry(Generic[T]):
    """Map stable configuration keys to executable class/function factories."""

    def __init__(self, name: str) -> None:
        if not name or name.strip() != name:
            raise ValueError("registry name must be a non-empty normalized path")
        self.name = name
        self._items: dict[str, Factory[T]] = {}

    def register(self, key: str) -> Callable[[Factory[T]], Factory[T]]:
        """Return a decorator that registers one factory exactly once."""

        if not key or key.strip() != key:
            raise ValueError("registry key must be a non-empty normalized string")

        def decorator(factory: Factory[T]) -> Factory[T]:
            if key in self._items:
                raise DuplicateRegistrationError(
                    f"[{self.name}] duplicate registration for {key!r}"
                )
            self._items[key] = factory
            return factory

        return decorator

    def require(self, key: str) -> Factory[T]:
        """Return a registered factory or fail without selecting a fallback."""

        try:
            return self._items[key]
        except KeyError as error:
            choices = ", ".join(repr(item) for item in self.keys()) or "<none>"
            raise UnknownRegistrationError(
                f"[{self.name}] unknown selection {key!r}; available: {choices}"
            ) from error

    def build(self, key: str, **kwargs: object) -> T:
        """Construct the selected implementation with the caller's section values."""

        return self.require(key)(**kwargs)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    @property
    def items(self) -> Mapping[str, Factory[T]]:
        return MappingProxyType(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._items)
