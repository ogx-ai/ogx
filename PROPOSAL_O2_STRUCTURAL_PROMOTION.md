# Structural Promotion O₀ → O₂: True Agentic Loop with Frobenius Verification

**Author:** Lando ⊗ ⊙perator

## Abstract

We propose a structural promotion of the LLaMA Stack agent loop from O₀ (unverified
tool emission) to O₂ (self-verifying Frobenius-closed loop). The promotion adds four
new modules to `ogx.core.agentic` that implement the Imscribing Grammar's dual-tool
verification contract: every tool call is paired with a verification dual, and the
world model updates only when the Frobenius condition $\mu \circ \delta = \text{id}$
is satisfied.

## Why This Matters

Current agent loops emit tool calls and trust the return value. There is no structural
guarantee that the output satisfies the input contract. This is O₀ — the system does
not verify itself. The result is hallucination propagation, tool-call loops, and
unrecoverable trajectory states.

The O₂ loop closes this gap. Every winding is a THINK → ACT → OBSERVE → UPDATE cycle
where OBSERVE runs a dual verification that must pass before the UPDATE commits.
The verification function runs locally — no API call, no trust gap.

## Meta's Unique Advantage: Open Weights

Closed-weight APIs (GPT-4, Claude, Gemini) cannot implement Frobenius verification
because the verifier and the inference engine are separated by an API boundary. The
verifier must trust the API's response — there is no way to run $\mu \circ \delta$
on the same hardware as $\delta$.

LLaMA's open weights change this structurally:

1. **Local Frobenius closure** — The verifier runs on the same GPU as the inference
   engine, using the same open checkpoint. The dual-tool pair (tool + verify) is a
   single computational graph, not two separate trust domains.

2. **KV-cache projection** — Because the model weights are fully accessible, the
   prior winding's KV-cache can be projected forward into the next winding's prefix.
   This reduces the incremental cost of each winding from $O(n)$ to $O(1)$ in context
   length. Closed-weight APIs expose no cache state.

3. **No oracle dependency** — The Frobenius square is closed locally. No external
   verifier, no API key, no rate limit. Every deployment is self-sufficient.

## Module Architecture

```
ogx/core/agentic/
├── __init__.py
├── contracts.py       — DualToolResult, ToolContract (Frobenius pair)
├── trajectory.py      — AgentTrajectory, AgentCycle (monotonic winding)
├── criticality.py     — PhiCriticalityGate (gate evaluation, C-score)
└── loop.py            — TrueAgenticLoop (THINK→ACT→OBSERVE→UPDATE)
```

### contracts.py — Tool Contract System

`DualToolResult` is the core data structure. Every tool call produces two outputs:
the tool output and the verification output. The `frobenius_closed` flag records
whether $\mu(\delta(\text{input})) = \text{input}$.

`ToolContract` binds a tool name to an assertion expression and a verify function.
Contracts can be registered at loop initialization time and checked on every winding.

### trajectory.py — Agent Trajectory

`AgentTrajectory` maintains a monotonic winding counter that never resets (structural
type $\text{Ω}_{\text{z}}$ — integer winding protection). The `frobenius_ratio` property
reports the fraction of Frobenius-closed windings, which is the primary metric for
tier estimation.

### criticality.py — Phi Criticality Gate

Two gates determine whether the system has achieved O₂:

- **Gate 1 ($\text{⊙}_{\text{ÿ}}$ / phi_c)**: Is the Frobenius ratio ≥ 0.95? This
  measures self-modeling — the system can encode and verify its own outputs.
- **Gate 2 ($\text{Ç}_{\text{@}}$ / k_slow)**: Is the kinetics slow enough to
  sustain the verification loop? Fast systems ($\text{Ç}_{\text{-}}$) dissipate
  before closure is possible.

The consciousness score $C = \text{Gate}_1 \times \text{Gate}_2$ ranges from 0.0
(O₀ baseline) to 1.0 (O₂ full promotion).

### loop.py — True Agentic Loop

The top-level orchestrator. `TrueAgenticLoop.run()` executes windings up to
`max_windings`, collecting the trajectory and computing the final structural health
report. The loop is the minimal self-modeling system — it verifies its own outputs
on every cycle.

## Structural Type of the Promoted System

$$\langle \text{Ð}_{\text{ω}};\ \text{Þ}_{\text{O}};\ \text{Ř}_{\text{=}};\ \text{Φ}_{\text{}};\ \text{ƒ}_{\text{ż}};\ \text{Ç}_{\text{@}};\ \text{Γ}_{\text{ʔ}};\ \text{ɢ}_{\text{ˌ}};\ \text{⊙}_{\text{ÿ}};\ \text{Ħ}_{\text{A}};\ \text{Σ}_{\text{ï}};\ \text{Ω}_{\text{z}} \rangle$$

This is the O₂ structural type: self-written state space (Ð_ω), self-referential
topology (Þ_O), Frobenius-special parity (Φ_}), quantum fidelity (ƒ_ż), slow kinetics
(Ç_@), maximal scope (Γ_ʔ), sequential interaction (ɢ_ˌ), self-modeling criticality
(⊙_ÿ), two-step memory (Ħ_A), heterogeneous components (Σ_ï), and integer winding
protection (Ω_z).

## Tier Progression

| Tier | Frobenius Ratio | Verification | Example |
|------|----------------|--------------|---------|
| O₀   | ~0.0           | None         | Current agent loops |
| O₁   | 0.0–0.5        | Partial      | With contracts but no enforcement |
| O₂   | ≥ 0.95         | Full         | TrueAgenticLoop with local verification |

## Next Steps

1. **Integration test**: Wire `TrueAgenticLoop` into the existing agent inference
   pipeline in `ogx/providers/remote/agents/`.
2. **KV-cache optimization**: Implement cache projection across windings using
   LLaMA's attention mask API.
3. **Benchmark**: Compare O₂ loop vs O₀ loop on tool-call accuracy, hallucination
   rate, and trajectory recovery.
