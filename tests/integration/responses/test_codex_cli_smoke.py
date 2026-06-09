# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Smoke test: drive the Codex CLI against the OGX Responses API.

This runs the real Codex CLI through `ogx connect codex --exec` so the test
exercises the same temporary CODEX_HOME, generated ogx.config.toml, and generated
model catalog that users get from the interactive command.

This runs live against a real backend, not in replay mode. The Codex CLI request
shape is produced outside this repo and is not stable enough to record by body
hash, so the CI workflow provisions Ollama and runs the smoke path live.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ogx.core.library_client import OGXAsLibraryClient

CODEX_CLI = shutil.which("codex")
EXPECTED_CODEX_OUTPUT = "OGX_CODEX_OK"

pytestmark = [
    pytest.mark.skipif(
        CODEX_CLI is None,
        reason="Codex CLI not installed; install @openai/codex to run",
    ),
    pytest.mark.skipif(
        os.getenv("OGX_ENABLE_CODEX_CLI_SMOKE") != "1",
        reason="Codex CLI smoke is live-only; set OGX_ENABLE_CODEX_CLI_SMOKE=1 to run",
    ),
]


def _build_minimal_env(tmp_path: Path) -> dict[str, str]:
    """Keep the real CLI away from the user's normal Codex/OpenAI config."""

    home = tmp_path / "home"
    tmp = tmp_path / "tmp"
    home.mkdir()
    tmp.mkdir()

    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "TMPDIR": str(tmp),
    }
    for key in ("LANG", "LC_ALL", "VIRTUAL_ENV"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def test_codex_cli_smoke_uses_generated_ogx_profile(ogx_client: Any, text_model_id: str, tmp_path: Path) -> None:
    """Codex completes one request through the generated OGX Responses profile."""

    if isinstance(ogx_client, OGXAsLibraryClient):
        pytest.skip("Codex CLI smoke test requires server mode")

    base_url = f"{ogx_client.base_url}/v1"
    prompt = f"Reply with exactly {EXPECTED_CODEX_OUTPUT} and nothing else. Do not use tools."

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ogx.cli.ogx",
            "connect",
            "codex",
            "--url",
            base_url,
            "--model",
            text_model_id,
            "--exec",
            prompt,
        ],
        cwd=Path.cwd(),
        env=_build_minimal_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"Failed to run Codex CLI through OGX: exit {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "ERROR:" not in combined_output, (
        f"Codex CLI reported an error while running through OGX\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.strip() == EXPECTED_CODEX_OUTPUT, (
        "Codex CLI did not return the expected final answer through OGX\n"
        f"expected: {EXPECTED_CODEX_OUTPUT}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
