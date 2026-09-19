"""Circuit factory."""

from __future__ import annotations

from typing import Any

from ai_qec.qec.circuits.base import QECCircuit
from ai_qec.qec.circuits.code_capacity import CodeCapacityExperiment
from ai_qec.qec.circuits.memory import MemoryExperiment
from ai_qec.qec.codes.base import QECCode


def build_circuit(config: dict[str, Any], code: QECCode) -> QECCircuit:
    """Build a QEC circuit description from experiment config."""
    qec_cfg = config.get("qec", {})
    task = str(qec_cfg.get("task", "memory"))
    rounds = int(qec_cfg.get("rounds", 1))
    # Return early when task == 'memory'.
    if task == "memory":
        return MemoryExperiment(code=code, rounds=rounds)
    # Follow this branch when task == 'code_capacity'.
    if task == "code_capacity":
        # Reject this state when rounds != 1.
        if rounds != 1:
            raise ValueError("code_capacity experiments require exactly one ideal syndrome round")
        return CodeCapacityExperiment(code=code)
    raise ValueError(f"Unknown QEC task/circuit: {task}")
