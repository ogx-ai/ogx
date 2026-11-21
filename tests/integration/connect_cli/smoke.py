# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared helpers for the `ogx connect <agent>` CLI smoke tests.

Each smoke test drives the real, unmodified agent CLI through the documented
zero-flag flow (an `ogx go --insecure --no-auth` server + `ogx connect
<agent>`) and asserts that one prompt round-trips:

- mock setups (``sentinel`` is set): the reply must contain the mock
  backend's fixed sentinel — a deterministic proof of the round-trip
- real-model setups: the run is asserted error-free; harnesses that report
  model usage may assert on it in their own test

These run live, never in replay mode: the CLIs bake the working directory,
date, and platform into every request body, so the request-body hashes the
recording system keys on are not reproducible across runs or machines.
"""

import dataclasses
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


def ogx_connect_cmd() -> list[str]:
    """The `ogx connect` entry point for subprocess use in the test environment."""
    return [sys.executable, "-m", "ogx.cli.ogx", "connect"]


def build_minimal_env(tmp_path: Path) -> dict[str, str]:
    """A scrubbed environment that keeps the agent CLI away from the user's real config.

    The agent's own config directories and provider credentials must not leak
    into the smoke run: connect generates the connection itself, and the
    launched CLI should only see what the test provides.
    """
    home = tmp_path / "home"
    tmp = tmp_path / "tmp"
    home.mkdir()
    tmp.mkdir()

    env = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "TMPDIR": str(tmp),
        "PATH": os.environ.get("PATH", ""),
    }
    for key in ("LANG", "LC_ALL", "VIRTUAL_ENV"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


@dataclasses.dataclass(frozen=True)
class Harness:
    """How one agent CLI participates in the shared smoke flow.

    ``agent_args`` are the harness's native flags; the prompt is appended
    after ``--``. ``extract_reply`` pulls the model's reply out of the CLI's
    stdout. ``error_markers`` are substrings that indicate a failure even
    when the process exits zero.
    """

    name: str
    binary: str
    agent_args: list[str]
    extract_reply: Callable[[str], str]
    error_markers: tuple[str, ...] = ()

    def skip_reason(self) -> str | None:
        if shutil.which(self.binary) is None:
            return f"{self.binary} CLI not installed"
        return None


def run_agent_smoke(
    harness: Harness,
    *,
    base_url: str,
    prompt: str,
    sentinel: str | None,
    model_id: str | None = None,
    tmp_path: Path | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    """Run `ogx connect <harness>` with one prompt and assert the round-trip.

    In mock mode (``sentinel`` set) the sentinel must appear in the extracted
    reply; otherwise the run is asserted error-free only. Returns the process
    so tests can make harness-specific assertions on the raw output.
    """
    # --model is a per-subcommand flag, so it goes after the agent name; the
    # native args and prompt are forwarded after --, exactly as the docs show.
    cmd = ogx_connect_cmd() + ["--url", base_url, harness.name]
    if model_id:
        cmd += ["--model", model_id]
    # -- marks the start of the native agent args, which (with the prompt) are
    # all forwarded verbatim after the ogx-level flags are parsed.
    cmd += ["--", *harness.agent_args, prompt]

    # Generous: the real CLIs make several large-context calls and a small
    # model on a CPU-only CI runner generates slowly (tens of seconds each).
    result = subprocess.run(
        cmd,
        cwd=tmp_path or Path.cwd(),
        env=build_minimal_env(tmp_path) if tmp_path else None,
        # The CLIs must never block on inherited stdin (codex exec reads
        # "additional input from stdin" when it is a TTY or pipe).
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    combined = f"{result.stdout}\n{result.stderr}"
    for marker in harness.error_markers:
        assert marker not in combined, (
            f"{harness.name} CLI reported an error while running through OGX\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert result.returncode == 0, (
        f"Failed to run {harness.name} via ogx connect: exit {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    reply = harness.extract_reply(result.stdout)
    if sentinel:
        assert sentinel in reply, (
            f"Mock sentinel {sentinel!r} not in {harness.name} reply; the request did not "
            f"round-trip through the backing model: {reply!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result
