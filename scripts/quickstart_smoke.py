#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Run the docs Quickstart the way a new user would, for every combination it offers.

The Quickstart page (docs/docs/getting_started/quickstart.mdx) renders its steps from
docs/docs/getting_started/quickstart/steps.json. This script reads the same file, selects
the steps for each supported combination of backend, server mode, and use case exactly as
the page does, and runs them: the shell commands as written, the Python files the page
shows, and the agents through `ogx connect`. The server starts from the latest PyPI release
or from locally built wheels, outside this repo's uv.lock.

Each step's optional `test` entry is the only place the run departs from the page:
  skip                 do not run the step (the reason is printed)
  requires_env         skip a combination whose backend keys are not in the environment
  server               start the step's command in the background and wait for `ready`
  capture_from_server  take a value the page asks the reader to copy from the server output
  fixture              files to place next to a Python snippet before running it
  agent_args           one-shot arguments that replace the agent's interactive session
  expect               a regex the step's output must match ({model} is the backend's model)

Usage:
  uv run --no-project python scripts/quickstart_smoke.py --list
  uv run --no-project python scripts/quickstart_smoke.py --backend ollama --server secure
  uv run --no-project python scripts/quickstart_smoke.py --wheels dist --backend ollama

Environment:
  QUICKSTART_OLLAMA_CMD  command that replaces `ollama` in the page's commands
                         (CI: "docker exec ollama ollama"; default: ollama)
"""

import argparse
import itertools
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
QUICKSTART_DIR = REPO_ROOT / "docs" / "docs" / "getting_started" / "quickstart"
STEPS_FILE = QUICKSTART_DIR / "steps.json"

SERVER_START_TIMEOUT_SECONDS = 900
STEP_TIMEOUT_SECONDS = 900
OGX_KEY_PATTERN = re.compile(r"ogk_[A-Za-z0-9_-]+")
EXPORT_PATTERN = re.compile(r"^export ([A-Z_][A-Z0-9_]*)=")

Selection = dict[str, str]


def _matches(when: dict[str, list[str]] | None, selection: Selection) -> bool:
    """Whether every dimension named in `when` is selected with one of the listed values.

    Mirrors matches() in docs/src/components/QuickstartSelector/index.jsx.
    """
    if not when:
        return True
    return all(selection.get(dimension) in values for dimension, values in when.items())


def _combinations(manifest: dict[str, Any]) -> list[Selection]:
    """Every selection the page allows, including unsupported ones."""
    selections: list[Selection] = [{}]
    for dimension in manifest["dimensions"]:
        expanded: list[Selection] = []
        for selection in selections:
            if not _matches(dimension.get("when"), selection):
                expanded.append(selection)
                continue
            for option in dimension["options"]:
                expanded.append({**selection, dimension["id"]: option["id"]})
        selections = expanded
    return selections


def _unsupported_reason(manifest: dict[str, Any], selection: Selection) -> str | None:
    for rule in manifest["unsupported"]:
        if _matches(rule["when"], selection):
            return str(rule["reason"])
    return None


def _model_for(manifest: dict[str, Any], selection: Selection) -> str:
    backend = next(d for d in manifest["dimensions"] if d["id"] == "backend")
    option = next(o for o in backend["options"] if o["id"] == selection["backend"])
    return str(option["model"])


def _substitute_model(manifest: dict[str, Any], text: str, model: str) -> str:
    """Swap the default model for the selected backend's, as the page does."""
    return text.replace(manifest["default_model"], model)


def _code_lines(step: dict[str, Any], selection: Selection) -> list[str]:
    lines: list[str] = []
    for entry in step.get("code", []):
        if isinstance(entry, str):
            lines.append(entry)
        elif _matches(entry.get("when"), selection):
            lines.append(entry["line"])
    return lines


def _label(selection: Selection) -> str:
    return " / ".join(selection[d] for d in ("backend", "server", "use_case", "agent") if d in selection)


def _redact(text: str) -> str:
    return OGX_KEY_PATTERN.sub("ogk_***", text)


def _run_with_timeout(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a step, reporting a timeout like `timeout` does: exit code 124 and the output so far."""
    try:
        return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=STEP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return subprocess.CompletedProcess(command, 124, output, f"Timed out after {STEP_TIMEOUT_SECONDS}s")


@dataclass
class Result:
    label: str
    status: str
    detail: str = ""


@dataclass
class Runner:
    manifest: dict[str, Any]
    ogx_command: list[str]
    ollama_command: list[str]
    results: list[Result] = field(default_factory=list)

    def _localize(self, command: str) -> str:
        """Point the page's `uvx ogx` and `ollama` at the install and Ollama under test."""
        command = re.sub(r"\buvx ogx\b", shlex.join(self.ogx_command), command)
        return re.sub(r"^ollama\b", shlex.join(self.ollama_command), command)

    def _run_shell(self, lines: list[str], env: dict[str, str], cwd: Path) -> tuple[int, str, dict[str, str]]:
        """Run lines in bash with `env` and return the exit code, output, and resulting environment."""
        env_dump = cwd / ".quickstart-env"
        script = "set -euo pipefail\n" + "\n".join(self._localize(line) for line in lines)
        script += f"\nenv -0 > {shlex.quote(str(env_dump))}\n"
        completed = _run_with_timeout(["bash", "-c", script], cwd, env)
        new_env = dict(env)
        if completed.returncode == 0 and env_dump.exists():
            entries = env_dump.read_text().split("\0")
            new_env = dict(entry.split("=", 1) for entry in entries if "=" in entry)
            env_dump.unlink()
        return completed.returncode, completed.stdout + completed.stderr, new_env

    def run_group(self, backend: str, server: str, selections: list[Selection], base_env: dict[str, str]) -> None:
        """Start one server for a backend and server mode, then run each use case against it."""
        steps = self.manifest["steps"]
        work_dir = Path(tempfile.mkdtemp(prefix="ogx-quickstart-"))
        server_log = work_dir / "server.log"
        server_process: subprocess.Popen[bytes] | None = None
        group_label = f"{backend} / {server}"

        missing_env = sorted(
            {
                name
                for selection in selections
                for step in steps
                if _matches(step.get("when"), selection)
                for name in step.get("test", {}).get("requires_env", [])
                if any(line.startswith(f"export {name}=") for line in _code_lines(step, selection))
                and not base_env.get(name)
            }
        )
        if missing_env:
            for selection in selections:
                self.results.append(Result(_label(selection), "SKIPPED", f"{', '.join(missing_env)} not set"))
            print(f"\n=== {group_label}: skipped, {', '.join(missing_env)} not set ===")
            return

        server_step = next(
            s
            for s in steps
            if s.get("phase") == "server" and "server" in s.get("test", {}) and _matches(s.get("when"), selections[0])
        )
        docker_image = server_step["test"]["server"].get("docker_image")
        try:
            print(f"\n=== {group_label}: setup ===")
            error = self._run_setup(selections, base_env, work_dir)
            if error is None:
                print(f"\n=== {group_label}: starting server ===")
                server_process, error = self._start_server(server_step, selections[0], base_env, work_dir, server_log)
            if error is None:
                for selection in selections:
                    self._run_client_steps(selection, base_env, work_dir, server_log)
            else:
                print(f"FAILED: {error}")
                if server_log.exists():
                    print(f"=== server output ===\n{_redact(server_log.read_text(errors='replace'))}")
                for selection in selections:
                    self.results.append(Result(_label(selection), "FAILED", error))
        finally:
            if server_process is not None and server_process.poll() is None:
                os.killpg(server_process.pid, signal.SIGINT)
                try:
                    server_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server_process.pid, signal.SIGKILL)
                    server_process.wait()
            if docker_image:
                containers = subprocess.run(
                    ["docker", "ps", "-q", "--filter", f"ancestor={docker_image}"], capture_output=True, text=True
                ).stdout.split()
                if containers:
                    subprocess.run(["docker", "rm", "-f", *containers], capture_output=True)
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_setup(self, selections: list[Selection], env: dict[str, str], work_dir: Path) -> str | None:
        """Run the setup steps of every combination sharing the server, each distinct step once."""
        seen: set[tuple[str, ...]] = set()
        for selection in selections:
            for step in self.manifest["steps"]:
                if step.get("phase") != "setup" or not _matches(step.get("when"), selection):
                    continue
                lines = _code_lines(step, selection)
                key = (step["id"], *lines)
                if key in seen:
                    continue
                seen.add(key)
                if "skip" in step.get("test", {}):
                    print(f"--- {step['id']}: skipped ({step['test']['skip']})")
                    continue
                print(f"--- {step['id']}: {'; '.join(lines)}")
                code, output, _ = self._run_shell(lines, env, work_dir)
                print(output.rstrip())
                if code != 0:
                    return f"Failed to run setup step {step['id']} (exit {code})"
        return None

    def _start_server(
        self, step: dict[str, Any], selection: Selection, env: dict[str, str], work_dir: Path, server_log: Path
    ) -> tuple[subprocess.Popen[bytes], str | None]:
        """Start the server step in the background and wait until its output shows it is ready."""
        command = "\n".join(self._localize(line) for line in _code_lines(step, selection))
        ready = re.compile(step["test"]["server"]["ready"])
        print(command)
        with server_log.open("wb") as log:
            process = subprocess.Popen(
                ["bash", "-c", "exec " + command],
                cwd=work_dir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
        while not ready.search(server_log.read_text(errors="replace")):
            if process.poll() is not None:
                return process, "Failed to start the server: it exited before it was ready"
            if time.monotonic() > deadline:
                return process, f"Failed to start the server within {SERVER_START_TIMEOUT_SECONDS}s"
            time.sleep(1)
        for line in server_log.read_text(errors="replace").splitlines():
            if "✓" in line or "✗" in line:
                print(_redact(line))
        return process, None

    def _run_client_steps(
        self, selection: Selection, base_env: dict[str, str], group_dir: Path, server_log: Path
    ) -> None:
        label = _label(selection)
        model = _model_for(self.manifest, selection)
        work_dir = Path(tempfile.mkdtemp(prefix="case-", dir=group_dir))
        env = dict(base_env)
        print(f"\n=== {label} ===")
        for step in self.manifest["steps"]:
            if step.get("phase") != "client" or not _matches(step.get("when"), selection):
                continue
            test = step.get("test", {})
            if "skip" in test:
                print(f"--- {step['id']}: skipped ({test['skip']})")
                continue
            if "file" in step:
                code, output = self._run_snippet(step, model, env, work_dir)
            elif "code" in step:
                lines = [_substitute_model(self.manifest, line, model) for line in _code_lines(step, selection)]
                for name, pattern in test.get("capture_from_server", {}).items():
                    if not any(line.startswith(f"export {name}=") for line in lines):
                        continue
                    captured = re.search(pattern, server_log.read_text(errors="replace"))
                    if captured is None:
                        self._fail(label, f"Failed to find {name} in the server output")
                        return
                    env[name] = captured.group(0)
                    lines = [line for line in lines if not line.startswith(f"export {name}=")]
                if "agent_args" in test:
                    lines = [line + " " + shlex.join(test["agent_args"][selection["agent"]]) for line in lines]
                print(f"--- {step['id']}: {_redact('; '.join(lines))}")
                code, output, env = self._run_shell(lines, env, work_dir)
            else:
                continue
            print(_redact(output.rstrip()))
            if code != 0:
                self._fail(label, f"Failed to run step {step['id']} (exit {code})")
                return
            expect = test.get("expect")
            if expect and not re.search(expect.replace("{model}", re.escape(model)), output):
                self._fail(label, f"Failed to find {expect!r} in the output of step {step['id']}")
                return
        self.results.append(Result(label, "PASSED"))

    def _run_snippet(self, step: dict[str, Any], model: str, env: dict[str, str], work_dir: Path) -> tuple[int, str]:
        """Run the page's Python file, with the backend's model, in a throwaway environment."""
        source = (QUICKSTART_DIR / step["file"]).read_text()
        (work_dir / step["file"]).write_text(_substitute_model(self.manifest, source, model))
        for name, fixture in step.get("test", {}).get("fixture", {}).items():
            shutil.copy(REPO_ROOT / fixture, work_dir / name)
        print(f"--- {step['id']}: python {step['file']}")
        completed = _run_with_timeout(
            [
                "uv",
                "run",
                "--quiet",
                "--no-project",
                "--with",
                ",".join(step.get("test", {}).get("deps", ["openai"])),
                "python",
                step["file"],
            ],
            work_dir,
            env,
        )
        return completed.returncode, completed.stdout + completed.stderr

    def _fail(self, label: str, message: str) -> None:
        print(f"FAILED: {message}")
        self.results.append(Result(label, "FAILED", message))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="Print every combination and whether it is supported")
    parser.add_argument("--wheels", type=Path, help="Run ogx from the ogx and ogx_api wheels in this directory")
    for dimension in ("backend", "server", "use-case", "agent"):
        parser.add_argument(
            f"--{dimension}",
            action="append",
            default=[],
            help=f"Only run combinations with this {dimension} (repeatable)",
        )
    return parser.parse_args()


def main() -> int:
    # Keep our progress lines in order with the step output when stdout is a CI log file.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    args = _parse_args()
    manifest = json.loads(STEPS_FILE.read_text())
    filters = {"backend": args.backend, "server": args.server, "use_case": args.use_case, "agent": args.agent}

    selections = [
        s for s in _combinations(manifest) if all(not values or s.get(dim) in values for dim, values in filters.items())
    ]
    if args.list:
        for selection in selections:
            reason = _unsupported_reason(manifest, selection)
            print(f"{_label(selection):45} {'unsupported' if reason else 'supported'}")
        return 0

    supported = [s for s in selections if _unsupported_reason(manifest, s) is None]
    if not supported:
        print("Failed to find a supported combination matching the filters", file=sys.stderr)
        return 1

    if args.wheels:
        wheels_dir = args.wheels.resolve()
        ogx_command = [
            "uvx",
            "--from",
            str(next(wheels_dir.glob("ogx-*.whl"))),
            "--with",
            str(next(wheels_dir.glob("ogx_api-*.whl"))),
            "ogx",
        ]
    else:
        ogx_command = ["uvx", "ogx"]
    runner = Runner(
        manifest=manifest,
        ogx_command=ogx_command,
        ollama_command=shlex.split(os.environ.get("QUICKSTART_OLLAMA_CMD", "ollama")),
    )

    base_env = dict(os.environ)
    for (backend, server), group in itertools.groupby(supported, key=lambda s: (s["backend"], s["server"])):
        runner.run_group(backend, server, list(group), base_env)

    print("\n=== Summary ===")
    for result in runner.results:
        print(f"{result.status:8} {result.label:45} {result.detail}")
    return 1 if any(r.status == "FAILED" for r in runner.results) else 0


if __name__ == "__main__":
    sys.exit(main())
