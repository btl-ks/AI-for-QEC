"""Domain adapter placeholder."""

from __future__ import annotations

from ai_qec.models.adapters.base import Adapter


class DomainAdapter(Adapter):
    """No-op domain adapter used to keep the V0.2 API shape visible."""

    def adapt(self) -> None:
        """Perform no adaptation until a concrete method is implemented."""
        return None
