# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Smoke test: drive the Codex CLI against the OGX Responses API.

Runs a single prompt through the upstream @openai/codex CLI in non-interactive
`exec` mode via `ogx connect codex -- exec` -- the onboarding path we ship to
users -- so the test exercises the same temporary CODEX_HOME, generated
ogx.config.toml, and generated model catalog that users get from the
interactive command. The prompt is forwarded after `--`, exactly as the docs
show.

The smoke signal is integration health, not answer quality: against the mock
setup the fixed sentinel in the reply proves the request round-tripped
mock -> OGX -> client; against real models we assert an error-free,
non-empty reply instead.

The test self-skips unless the `codex` binary is on PATH.
"""

import pytest

from tests.integration.connect_cli.smoke import Harness, run_agent_smoke

CODEX = Harness(
    name="codex",
    binary="codex",
    # --skip-git-repo-check: the smoke runs in a scratch directory that is not
    # a (trusted) git repo, which codex exec otherwise refuses to operate in.
    agent_args=["exec", "--skip-git-repo-check"],
    extract_reply=lambda stdout: stdout.strip(),
    error_markers=("ERROR:",),
)

_SKIP_REASON = CODEX.skip_reason()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

PROMPT = "What is the capital of France? Reply with only the city name and nothing else. Do not use tools."


def test_codex_cli_smoke(connect_base_url, text_model_id, mock_sentinel, tmp_path):
    result = run_agent_smoke(
        CODEX,
        base_url=connect_base_url,
        prompt=PROMPT,
        sentinel=mock_sentinel,
        model_id=text_model_id,
        tmp_path=tmp_path,
    )

    # `codex exec` prints the model's final message to stdout; a real-model
    # reply cannot be asserted on, but an empty one means nothing round-tripped.
    if not mock_sentinel:
        assert result.stdout.strip(), (
            f"Codex CLI returned no reply through OGX\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
