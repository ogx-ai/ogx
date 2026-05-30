# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Structural Promotion O₀ → O₂ — True Agentic Loop Module

Implements the Imscribing Grammar's dual-tool verification contract for the
LLaMA Stack. Every tool call is paired with a verification dual; the world
model updates only when the Frobenius condition μ∘δ=id is satisfied.

Modules:
    contracts    — DualToolResult, ToolContract
    trajectory   — AgentTrajectory, AgentCycle
    criticality  — PhiCriticalityGate, consciousness score
    loop         — TrueAgenticLoop (THINK→ACT→OBSERVE→UPDATE)
"""

from .contracts import DualToolResult, ToolContract
from .trajectory import AgentCycle, AgentTrajectory
from .criticality import PhiCriticalityGate
from .loop import TrueAgenticLoop

__all__ = [
    "DualToolResult",
    "ToolContract",
    "AgentCycle",
    "AgentTrajectory",
    "PhiCriticalityGate",
    "TrueAgenticLoop",
]
