"""Pauli-frame state."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PauliFrame:
    """Collect Pauli-frame updates produced during decoding."""
    updates: list[str] = field(default_factory=list)

    def append(self, update: str) -> None:
        """Append one Pauli-frame update token."""
        self.updates.append(update)
