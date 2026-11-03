# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Smoke test: drive the Claude Code CLI against the OGX Messages API.

Runs a single prompt through the upstream @anthropic-ai/claude-code CLI via
`ogx connect claude` -- the onboarding path we ship to users -- against a
local OGX server. This exercises the real client end-to-end against
/v1/messages: the CLI emits its full system prompt, tool definitions, and an
inline system-role message, all of which the server must accept and dispatch
to the backing provider before the model's reply comes back.

This runs LIVE against a real backend, not in replay mode. The CLI bakes the
working directory, date, and platform into every request body, so the
request-body hashes the recording system keys on are not reproducible across
runs or machines; recording/replay is therefore not viable for the real CLI.

The test self-skips unless the `claude` binary is on PATH.
"""

import json
import shutil
import subprocess

import pytest

CLAUDE_CLI = shutil.which("claude")
OGX_CLI = shutil.which("ogx")

pytestmark = pytest.mark.skipif(
    CLAUDE_CLI is None or OGX_CLI is None,
    reason="the claude-code CLI and ogx must both be on PATH to run",
)


def test_claude_code_cli_smoke(messages_base_url, text_model_id, mock_sentinel, tmp_path):
    """Claude Code CLI completes a request against /v1/messages without error.

    The smoke signal is integration health, not answer quality: the CLI sends
    its full agentic payload (system prompt, tools, inline system message) to
    OGX, OGX routes it to the backing model, and the session completes with no
    API error. Against the mock setup the fixed sentinel in the reply proves
    the request round-tripped mock -> OGX -> client; against real models we
    cannot assert on output, so we confirm the model was used instead. A
    regression like a rejected system-role message (which would surface as an
    API error) is caught either way.
    """
    prompt = "What is the capital of France? Reply with only the city name and nothing else."

    # `ogx connect claude` builds the ANTHROPIC_* environment from the group
    # --url flag and --model and execs the CLI, passing through its exit code.
    # The test passes no thinking configuration: the adapter is what makes this
    # work for users, so the test exercises exactly the shipped path.
    result = subprocess.run(
        [
            OGX_CLI,
            "connect",
            "--url",
            str(messages_base_url).rstrip("/") + "/v1",
            "claude",
            "--model",
            text_model_id,
            "--",
            "--print",
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
            prompt,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        # Generous: the real CLI makes several large-context calls and a small
        # model on a CPU-only CI runner generates slowly (tens of seconds each).
        timeout=600,
    )

    assert result.returncode == 0, (
        f"Failed to run Claude Code CLI via ogx connect: exit {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    data = json.loads(result.stdout)
    assert data.get("is_error") is False and data.get("subtype") == "success", (
        f"Claude Code CLI reported an error talking to /v1/messages: {json.dumps(data, indent=2)}"
    )
    # Confirm the request actually reached the backing model through /v1/messages.
    if mock_sentinel:
        assert mock_sentinel in (data.get("result") or ""), (
            f"Mock sentinel {mock_sentinel!r} not in CLI result; the request did not round-trip "
            f"through the backing model: {json.dumps(data, indent=2)}"
        )
    else:
        assert text_model_id in data.get("modelUsage", {}), (
            f"Expected model {text_model_id} in CLI modelUsage; got: {json.dumps(data.get('modelUsage', {}))}"
        )
