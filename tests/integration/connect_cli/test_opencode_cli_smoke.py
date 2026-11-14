# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Smoke test: drive the OpenCode CLI against the OGX OpenAI-compatible APIs.

Runs a single prompt through the upstream opencode CLI (@opencode/cli) in
non-interactive `run` mode via `ogx connect opencode -- run --standalone` --
the onboarding path we ship to users -- so the test exercises the same
generated OPENCODE_CONFIG_CONTENT provider config that users get from the
interactive command. The prompt is forwarded after `--`, exactly as the docs
show.

The smoke signal is integration health, not answer quality: against the mock
setup the fixed sentinel in the reply proves the request round-tripped
mock -> OGX -> client; against real models we assert an error-free,
non-empty reply instead.

The test self-skips unless an opencode v2 binary is on PATH (v1 lacks
`--standalone` and would fail rather than run).
"""

import re
import subprocess

import pytest

from tests.integration.connect_cli.smoke import Harness, run_agent_smoke

OPENCODE = Harness(
    name="opencode",
    binary="opencode",
    # --standalone: run with a private server instead of the shared
    # background service, so each smoke invocation is self-contained and
    # cannot reuse a daemon started by an earlier run with a different env.
    agent_args=["run", "--standalone"],
    extract_reply=lambda stdout: stdout.strip(),
    error_markers=("Error:",),
)


def _skip_reason() -> str | None:
    reason = OPENCODE.skip_reason()
    if reason is not None:
        return reason
    try:
        out = subprocess.run([OPENCODE.binary, "--version"], capture_output=True, text=True, timeout=30)
        match = re.search(r"(\d+)\.\d+\.\d+", out.stdout or out.stderr)
    except (OSError, subprocess.SubprocessError):
        return "Failed to determine opencode version"
    if match is None or int(match.group(1)) != 2:
        return "opencode v2 (@opencode/cli) not found on PATH; the smoke targets v2"
    return None


_SKIP_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

PROMPT = "What is the capital of France? Reply with only the city name and nothing else. Do not use tools."


def test_opencode_cli_smoke(connect_base_url, text_model_id, mock_sentinel, tmp_path):
    result = run_agent_smoke(
        OPENCODE,
        base_url=connect_base_url,
        prompt=PROMPT,
        sentinel=mock_sentinel,
        model_id=text_model_id,
        tmp_path=tmp_path,
    )

    # `opencode run` prints the model's reply to stdout; a real-model reply
    # cannot be asserted on, but an empty one means nothing round-tripped.
    if not mock_sentinel:
        assert result.stdout.strip(), (
            f"OpenCode CLI returned no reply through OGX\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
