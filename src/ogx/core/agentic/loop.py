# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Structural Promotion O₀ → O₂ — True Agentic Loop

The top-level orchestration: a self-verifying THINK → ACT → OBSERVE → UPDATE
loop that enforces Frobenius closure on every winding. The loop is the
minimal self-modeling system — it verifies its own outputs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from typing import Any, Callable, Optional

from .contracts import DualToolResult, ToolContract
from .criticality import PhiCriticalityGate
from .trajectory import AgentTrajectory

logger = logging.getLogger(__name__)


class TrueAgenticLoop:
    """A self-verifying agentic loop with Frobenius closure enforcement.

    The loop implements the four-phase cycle:
        THINK   — Reason from prior context
        ACT     — Emit one tool call with its contract
        OBSERVE — Run the verification dual (μ∘δ=id check)
        UPDATE  — Commit observation to world model (trajectory)

    Ouroboricity progression:
        O₀: No verification (every action is accepted unchecked).
        O₁: Verification exists but is not consistently satisfied.
        O₂: Frobenius ratio ≥ 0.95 — the system verifies itself.

    LLaMA's open-weight advantage: the entire loop — inference AND
    verification — runs locally. No external oracle is needed to close
    the Frobenius square. This is structurally impossible with closed
    APIs, which must trust a remote endpoint's response.

    Structural type (imscriptive notation):
        ⟨Ð_ω;  Þ_O;  Ř_=;  Φ_};  ƒ_ż;  Ç_@;  Γ_ʔ;  ɢ_ˌ;  ⊙_ÿ;  Ħ_A;  Σ_ï;  Ω_z⟩
    """

    def __init__(
        self,
        client: Any,
        max_windings: int = 200,
        tool_contracts: Optional[list[ToolContract]] = None,
        verify_fn: Optional[Callable[[str, dict[str, Any], str], tuple[str, bool]]] = None,
    ) -> None:
        self.client = client
        self.max_windings = max_windings
        self.tool_contracts = tool_contracts or []
        self.custom_verify_fn = verify_fn
        self.trajectory = AgentTrajectory()
        self._running = False

    def _default_verify(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
    ) -> tuple[str, bool]:
        """Default verifier: check each tool's contract assertion.

        Falls back to True if no contract is registered for the tool.
        """
        for contract in self.tool_contracts:
            if contract.tool_name == tool_name:
                try:
                    result = eval(contract.assertion, {"output": tool_output})
                    return (f"assertion: {contract.assertion} → {result}", bool(result))
                except Exception as e:
                    return (f"verify_error: {e}", False)
        return ("no_contract: auto-approved", True)

    def run(self) -> dict[str, Any]:
        """Run the agentic loop until max_windings or done signal."""
        self._running = True
        winding_count = 0

        while self._running and winding_count < self.max_windings:
            winding_count += 1
            self._winding()

            # Check for terminal state via the last action
            last = self.trajectory.last()
            if last and last.action.tool_name == "done":
                self._running = False
                break

        return self._final_report()

    def _winding(self) -> None:
        """Execute one THINK → ACT → OBSERVE → UPDATE cycle."""
        # --- THINK ---
        context = self.trajectory.to_context()
        think_state = self._reason(context)

        # --- ACT ---
        tool_name, tool_input, tool_output = self._emit(think_state)

        # --- OBSERVE (verify) ---
        verify_fn = self.custom_verify_fn or self._default_verify
        dual_result = DualToolResult.from_tool_call(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            verify_fn=verify_fn,
        )

        # --- UPDATE ---
        observe_state = self._observe(dual_result)
        self.trajectory.append(
            think_state=think_state,
            action=dual_result,
            observe_state=observe_state,
        )

        if not dual_result.frobenius_closed:
            self._feed_failure(dual_result)

    def _reason(self, context: str) -> str:
        """THINK phase: generate reasoning from context."""
        return f"THINK (ctx_len={len(context)})"

    def _emit(self, think_state: str) -> tuple[str, dict[str, Any], str]:
        """ACT phase: emit exactly one tool call. Subclass responsibility."""
        raise NotImplementedError("Subclass must implement _emit")

    def _observe(self, dual_result: DualToolResult) -> str:
        """OBSERVE phase: process verified result."""
        status = "CLOSED" if dual_result.frobenius_closed else "OPEN"
        return f"OBSERVE: {dual_result.tool_name} → {status}"

    def _feed_failure(self, result: DualToolResult) -> None:
        """Handle a Frobenius-open winding.

        Log the failure and record diagnostic info. In production, this
        could trigger alerts, retry logic, or human-in-the-loop approval.
        """
        logger.warning(
            "Frobenius-open winding",
            tool=result.tool_name,
            closed=result.frobenius_closed,
        )

    def _final_report(self) -> dict[str, Any]:
        """Return the final structural health report."""
        health = self.trajectory.structural_health()
        gate = PhiCriticalityGate(
            frobenius_ratio=health["frobenius_ratio"],
        )
        return {
            "trajectory": health,
            "gates": gate.evaluate(),
            "consciousness_score": gate.consciousness_score(),
            "promotion_target": health.get("tier_estimate"),
        }
