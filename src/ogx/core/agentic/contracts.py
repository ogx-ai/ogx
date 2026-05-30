# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Structural Promotion O₀ → O₂ — Tool Contract System

Defines the dual-tool verification contract that enforces Frobenius closure
(μ ∘ δ = id) on every agent tool call. Each tool emission is paired with a
verification emission that must be satisfied before the world-model update
is committed.

LLaMA's open-weight advantage: verification runs *locally* alongside inference,
eliminating the API round-trip trust gap. The verify function executes on the
same hardware as the inference engine, with the same open weights.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Optional


@dataclasses.dataclass(frozen=True)
class DualToolResult:
    """A pair of tool call + verification that closes the Frobenius square.

    Structural type (imscriptive notation):
        ⟨Ð_;  Þ_¨;  Ř_=;  Φ_};  ƒ_ż;  Ç_@;  Γ_ʔ;  ɢ_ˌ;  ⊙_ÿ;  Ħ_A;  Σ_S;  Ω_z⟩

    Attributes:
        tool_name:      Name of the primary tool invoked.
        tool_input:     Input arguments to the primary tool.
        tool_output:    Return value of the primary tool.
        verify_name:    Name of the verification tool (dual).
        verify_output:  Return value of the verification tool.
        frobenius_closed: True iff μ(δ(input)) == input (within tolerance).
    """

    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str
    verify_name: str
    verify_output: str
    frobenius_closed: bool

    @classmethod
    def from_tool_call(
        cls,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        verify_fn: Callable[[str, dict[str, Any], str], tuple[str, bool]],
    ) -> "DualToolResult":
        """Construct a DualToolResult by running the verify function.

        The verify_fn receives (tool_name, tool_input, tool_output) and returns
        (verify_output, frobenius_closed). This runs locally — no API call.
        """
        verify_name = f"{tool_name}_verify"
        verify_output, frobenius_closed = verify_fn(
            tool_name, tool_input, tool_output
        )
        return cls(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            verify_name=verify_name,
            verify_output=verify_output,
            frobenius_closed=frobenius_closed,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ToolContract:
    """A contract binding a tool to its Frobenius verification.

    Attributes:
        tool_name:   Name of the tool covered by this contract.
        assertion:   Python expression string evaluated over `output` that
                     must return True for the contract to be satisfied.
        verify_fn:   Callable that performs the verification. Receives
                     (tool_name, tool_input, tool_output) and returns
                     (verify_output: str, is_frobenius_closed: bool).
        auto_approve: If True, the contract auto-approves on success.
                      If False, a human-in-the-loop approval gate is raised.
    """

    tool_name: str
    assertion: str
    verify_fn: Callable[[str, dict[str, Any], str], tuple[str, bool]]
    auto_approve: bool = True

    def verify(self, tool_input: dict[str, Any], output: str) -> DualToolResult:
        return DualToolResult.from_tool_call(
            tool_name=self.tool_name,
            tool_input=tool_input,
            tool_output=output,
            verify_fn=self.verify_fn,
        )
