"""Continual learning placeholder."""

from __future__ import annotations


def update_stream_state(state: dict, batch_id: str) -> dict:
    """Record the most recently processed streaming batch id."""
    state = dict(state)
    state["last_batch_id"] = batch_id
    return state
