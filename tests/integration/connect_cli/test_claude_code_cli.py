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

The smoke signal is integration health, not answer quality: against the mock
setup the fixed sentinel in the reply proves the request round-tripped
mock -> OGX -> client; against real models we cannot assert on output, so we
confirm the model was used instead. A regression like a rejected system-role
message (which would surface as an API error) is caught either way.

The test self-skips unless the `claude` binary is on PATH.
"""

import json

import pytest

from tests.integration.connect_cli.smoke import Harness, run_agent_smoke

CLAUDE = Harness(
    name="claude",
    binary="claude",
    agent_args=["--print", "--output-format", "json", "--dangerously-skip-permissions"],
    extract_reply=lambda stdout: json.loads(stdout).get("result") or "",
)

_SKIP_REASON = CLAUDE.skip_reason()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

PROMPT = "What is the capital of France? Reply with only the city name and nothing else."


def test_claude_code_cli_smoke(connect_base_url, text_model_id, mock_sentinel, tmp_path):
    # `ogx connect claude` builds the ANTHROPIC_* environment from the group
    # --url flag and --model and execs the CLI, passing through its exit code.
    # The test passes no thinking configuration: the adapter is what makes this
    # work for users, so the test exercises exactly the shipped path.
    result = run_agent_smoke(
        CLAUDE,
        base_url=connect_base_url,
        prompt=PROMPT,
        sentinel=mock_sentinel,
        model_id=text_model_id,
        tmp_path=tmp_path,
    )

    data = json.loads(result.stdout)
    assert data.get("is_error") is False and data.get("subtype") == "success", (
        f"Claude Code CLI reported an error talking to /v1/messages: {json.dumps(data, indent=2)}"
    )
    # Against real models the reply cannot be asserted on, so confirm the
    # request actually reached the backing model through /v1/messages.
    if not mock_sentinel:
        assert text_model_id in data.get("modelUsage", {}), (
            f"Expected model {text_model_id} in CLI modelUsage; got: {json.dumps(data.get('modelUsage', {}))}"
        )
