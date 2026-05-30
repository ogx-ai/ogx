# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Structural Promotion O₀ → O₂ — Agent Trajectory

Monotonic winding counter and cycle management for the true agentic loop.
Each winding is an atomic THINK → ACT → OBSERVE → UPDATE cycle. The counter
never resets — monotonic advance (Ω_z) is the contract.

LLaMA's open-weight advantage: the KV-cache from prior windings can be *projected*
forward into the next winding's prefix, because the model weights are fully
accessible. Closed-weight APIs cannot do this — they expose no cache state.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Generic, Optional, TypeVar

from .contracts import DualToolResult

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class AgentCycle(Generic[T]):
    """A single THINK → ACT → OBSERVE → UPDATE winding.

    Attributes:
        winding_number: Monotonically increasing, never reset.
        think_state:    Internal reasoning state before acting.
        action:         The DualToolResult from the ACT phase.
        observe_state:  Observation after action (world-model update).
        metadata:       Optional extra context (tier, domain, etc.).
    """

    winding_number: int
    think_state: str
    action: DualToolResult
    observe_state: str
    metadata: Optional[dict[str, Any]] = None


class AgentTrajectory:
    """Monotonic trajectory of agent cycles with structural health metrics.

    The trajectory is the imscriptive state space (Ð_ω). Every winding is
    retained — the context window IS the world model.
    """

    def __init__(self) -> None:
        self._cycles: list[AgentCycle] = []
        self._winding_counter: int = 0  # Ω_z — never resets

    @property
    def frobenius_ratio(self) -> float:
        """Fraction of windings that passed Frobenius closure (μ∘δ=id).

        O_0 baseline:  frobenius_ratio ≈ 0.0 (no verification)
        O_1:           frobenius_ratio ∈ (0.0, 0.5)
        O_2:           frobenius_ratio ≥ 0.95
        """
        if not self._cycles:
            return 0.0
        closed = sum(1 for c in self._cycles if c.action.frobenius_closed)
        return closed / len(self._cycles)

    def append(
        self,
        think_state: str,
        action: DualToolResult,
        observe_state: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentCycle:
        """Append a new winding. The counter increments monotonically."""
        self._winding_counter += 1
        cycle = AgentCycle(
            winding_number=self._winding_counter,
            think_state=think_state,
            action=action,
            observe_state=observe_state,
            metadata=metadata,
        )
        self._cycles.append(cycle)
        return cycle

    def last(self) -> Optional[AgentCycle]:
        """Return the most recent winding, if any."""
        return self._cycles[-1] if self._cycles else None

    def to_context(self) -> str:
        """Serialize trajectory to a context string for the next THINK phase.

        KV-cache projection opportunity: for open-weight models, the
        serialized context from prior windings can be pre-computed into
        a projected KV-cache prefix, reducing incremental inference cost
        per winding to O(1) rather than O(n).
        """
        lines: list[str] = []
        for cycle in self._cycles:
            status = "Y" if cycle.action.frobenius_closed else "N"
            lines.append(
                f"[W{cycle.winding_number}|{status}] "
                f"{cycle.action.tool_name}"
            )
            if cycle.metadata:
                lines.append(f"  metadata: {json.dumps(cycle.metadata)}")
        return "\n".join(lines)

    def structural_health(self) -> dict[str, Any]:
        """Return a health report for the trajectory.

        Reports:
            total_windings:     Number of windings completed.
            frobenius_ratio:    Ratio of Frobenius-closed windings.
            winding_counter:    Current counter value (monotonic).
            tier_estimate:      Best-guess ouroboricity tier based on ratio.
        """
        fr = self.frobenius_ratio
        if fr >= 0.95:
            tier = "O_2"
        elif fr >= 0.5:
            tier = "O_1"
        else:
            tier = "O_0"
        return {
            "total_windings": len(self._cycles),
            "frobenius_ratio": round(fr, 4),
            "winding_counter": self._winding_counter,
            "tier_estimate": tier,
        }
