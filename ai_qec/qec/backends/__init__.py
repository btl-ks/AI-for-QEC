"""Interfaces implemented by concrete QEC execution backends."""

from .protocol import BackendCompatibility, QECBackend

__all__ = ["BackendCompatibility", "QECBackend"]
