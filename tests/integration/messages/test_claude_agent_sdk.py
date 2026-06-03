# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Smoke test: drive the Claude Agent SDK against the OGX Messages API.

Points the SDK at a local OGX server and runs a single prompt through the
upstream `claude-agent-sdk` Python package (github.com/anthropics/
claude-agent-sdk-python). The SDK does not speak HTTP itself: it spawns the
Claude Code CLI as a subprocess and parses its streamed session output. This
exercises a different client surface than the CLI smoke test -- the SDK's
session machinery (message streaming, ResultMessage parsing) on top of the same
end-to-end path through /v1/messages: full system prompt, tool definitions, and
an inline system-role message that the server must accept and dispatch to the
backing provider.

This runs LIVE against a real backend, not in replay mode. The SDK drives the
real CLI, which bakes the working directory, date, and platform into every
request body, so the request-body hashes the recording system keys on are not
reproducible across runs or machines; recording/replay is therefore not viable.

The test self-skips unless both the `claude-agent-sdk` package and the `claude`
binary are available, since the SDK requires the CLI at runtime.
"""

import asyncio
import importlib.util
import shutil

import pytest

CLAUDE_CLI = shutil.which("claude")
HAS_SDK = importlib.util.find_spec("claude_agent_sdk") is not None

pytestmark = pytest.mark.skipif(
    CLAUDE_CLI is None or not HAS_SDK,
    reason="claude-agent-sdk and the claude CLI must both be installed to run",
)


def _run_query(prompt: str, base_url: str, model: str, cwd: str) -> list:
    """Run a single Agent SDK query() to completion and return all messages."""
    from claude_agent_sdk import ClaudeAgentOptions, query

    options = ClaudeAgentOptions(
        model=model,
        # Run from an isolated directory so the spawned CLI does not pick up
        # repo-local context, which matters while permissions are bypassed.
        cwd=cwd,
        # Passed to the spawned CLI subprocess so it reaches OGX instead of
        # api.anthropic.com.
        env={
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_API_KEY": "dummy",
            "ANTHROPIC_MODEL": model,
        },
        # The prompt is pure Q&A and triggers no tools, but bypass permissions
        # so the non-interactive session can never stall on a permission prompt.
        permission_mode="bypassPermissions",
    )

    messages: list = []

    async def _collect() -> None:
        async for message in query(prompt=prompt, options=options):
            messages.append(message)

    # Generous: the SDK drives the real CLI, which makes several large-context
    # calls, and a small model on a CPU-only CI runner generates slowly (tens of
    # seconds each). Bound it so a hung session fails loudly instead of riding
    # the job timeout.
    asyncio.run(asyncio.wait_for(_collect(), timeout=600))
    return messages


def test_claude_agent_sdk_smoke(messages_base_url, text_model_id, tmp_path):
    """Claude Agent SDK completes a session against /v1/messages without error.

    The smoke signal is integration health, not answer quality: the SDK drives a
    full agentic session (system prompt, tools, inline system message) against
    OGX, OGX routes it to the backing model, and the session terminates with a
    successful ResultMessage. We deliberately do not assert on the model's text
    output -- a small local model driving the Claude Code harness cannot be
    relied on to produce a specific answer, but a regression like a rejected
    system-role message (which would surface as a session error) is caught here.
    """
    from claude_agent_sdk import ResultMessage

    prompt = "What is the capital of France? Reply with only the city name and nothing else."
    base_url = str(messages_base_url).rstrip("/")

    messages = _run_query(prompt, base_url, text_model_id, cwd=str(tmp_path))

    results = [m for m in messages if isinstance(m, ResultMessage)]
    assert results, f"Agent SDK session produced no ResultMessage; got: {[type(m).__name__ for m in messages]}"

    result = results[-1]
    assert result.subtype == "success" and not result.is_error, (
        f"Agent SDK session reported an error talking to /v1/messages: "
        f"subtype={result.subtype} is_error={result.is_error} errors={result.errors}"
    )
    # Confirm the request actually reached the backing model through /v1/messages.
    model_usage = result.model_usage or {}
    assert text_model_id in model_usage, (
        f"Expected model {text_model_id} in ResultMessage.model_usage; got: {list(model_usage)}"
    )
