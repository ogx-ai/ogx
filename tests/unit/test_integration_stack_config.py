# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""tests/integration/stack_config.py is the single parser for --stack-config / OGX_CONFIG,
shared by the pytest client fixture, test gating and scripts/integration-tests.sh (#7072)."""

import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.integration.fixtures import common
from tests.integration.stack_config import (
    SERVER_MODE_ENV_DEFAULTS,
    StackConfigKind,
    apply_server_mode_env_defaults,
    main,
    parse_stack_config,
    resolve_stack_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STACK_CONFIG_SCRIPT = REPO_ROOT / "tests" / "integration" / "stack_config.py"


class TestParseStackConfig:
    @pytest.mark.parametrize(
        ("value", "kind", "target", "port"),
        [
            ("ci-tests", StackConfigKind.LIBRARY, "ci-tests", None),
            ("path/to/config.yaml", StackConfigKind.LIBRARY, "path/to/config.yaml", None),
            ("ci-tests::run.yaml", StackConfigKind.LIBRARY, "ci-tests::run.yaml", None),
            (
                "inference=remote::ollama,vector_io=inline::faiss",
                StackConfigKind.DYNAMIC,
                "inference=remote::ollama,vector_io=inline::faiss",
                None,
            ),
            ("server:ci-tests", StackConfigKind.SERVER, "ci-tests", None),
            ("server:ci-tests:9000", StackConfigKind.SERVER, "ci-tests", 9000),
            ("server:ci-tests::run.yaml", StackConfigKind.SERVER, "ci-tests::run.yaml", None),
            ("docker:ci-tests", StackConfigKind.DOCKER, "ci-tests", None),
            ("http://localhost:8321", StackConfigKind.URL, "http://localhost:8321", None),
            ("https://example.com", StackConfigKind.URL, "https://example.com", None),
        ],
    )
    def test_classifies_every_form(self, value, kind, target, port):
        parsed = parse_stack_config(value)

        assert (parsed.kind, parsed.target, parsed.port) == (kind, target, port)

    @pytest.mark.parametrize(
        ("value", "is_server"),
        [
            ("ci-tests", False),
            ("inference=remote::ollama", False),
            ("server:ci-tests", True),
            ("docker:ci-tests", True),
            ("http://localhost:8321", True),
        ],
    )
    def test_is_server(self, value, is_server):
        assert parse_stack_config(value).is_server is is_server

    def test_a_library_config_starting_with_http_is_not_a_url(self):
        """The old prefix check classified anything starting with "http" as a server."""
        assert parse_stack_config("http-distro").kind == StackConfigKind.LIBRARY

    @pytest.mark.parametrize("value", ["", "   ", "server:", "server::9000", "docker:", "server:ci:abc", "server:ci:0"])
    def test_rejects_malformed_values(self, value):
        with pytest.raises(ValueError, match="Failed to parse stack config"):
            parse_stack_config(value)

    def test_server_port_prefers_the_explicit_suffix(self):
        assert parse_stack_config("server:ci-tests:9000").server_port({"OGX_PORT": "1234"}) == 9000

    def test_server_port_falls_back_to_ogx_port_then_the_default(self):
        parsed = parse_stack_config("server:ci-tests")

        assert parsed.server_port({"OGX_PORT": "1234"}) == 1234
        assert parsed.server_port({}) == 8321

    @pytest.mark.parametrize(
        ("value", "deps_args"),
        [
            ("ci-tests", ["ci-tests"]),
            ("server:ci-tests:9000", ["ci-tests"]),
            ("inference=remote::ollama", ["--providers", "inference=remote::ollama"]),
            ("docker:ci-tests", None),
            ("http://localhost:8321", None),
        ],
    )
    def test_deps_args_resolve_dynamic_specs_through_providers(self, value, deps_args):
        assert parse_stack_config(value).deps_args() == deps_args


class TestResolveStackConfig:
    def test_option_wins_over_ogx_config(self):
        resolved = resolve_stack_config("ci-tests", {"OGX_CONFIG": "server:ci-tests"})

        assert resolved is not None and resolved.kind == StackConfigKind.LIBRARY

    def test_falls_back_to_ogx_config(self):
        resolved = resolve_stack_config(None, {"OGX_CONFIG": "server:ci-tests"})

        assert resolved is not None and resolved.kind == StackConfigKind.SERVER

    def test_neither_is_none(self):
        assert resolve_stack_config(None, {}) is None


class TestShellOutputForTheRunnerScript:
    def _eval(self, value: str) -> dict[str, str]:
        """Run the module as the runner script does and evaluate its output in bash."""
        script = (
            f'eval "$({shlex.quote(sys.executable)} -I {shlex.quote(str(STACK_CONFIG_SCRIPT))} shell {shlex.quote(value)})"; '
            'printf "%s\\n" "$STACK_CONFIG_KIND" "$STACK_CONFIG_TARGET" "$STACK_CONFIG_PORT" '
            '"$STACK_CONFIG_IS_SERVER" "${STACK_CONFIG_DEPS_ARGS[*]}"'
        )
        output = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
        kind, target, port, is_server, deps = output.splitlines()
        return {"kind": kind, "target": target, "port": port, "is_server": is_server, "deps": deps}

    def test_server_with_port(self):
        assert self._eval("server:ci-tests:9000") == {
            "kind": "server",
            "target": "ci-tests",
            "port": "9000",
            "is_server": "true",
            "deps": "ci-tests",
        }

    def test_dynamic_spec_uses_providers_for_the_dependency_preflight(self):
        result = self._eval("inference=remote::ollama,vector_io=inline::faiss")

        assert result["kind"] == "dynamic"
        assert result["deps"] == "--providers inference=remote::ollama,vector_io=inline::faiss"

    def test_docker_has_no_local_dependency_preflight(self):
        result = self._eval("docker:ci-tests")

        assert (result["kind"], result["is_server"], result["deps"]) == ("docker", "true", "")

    def test_malformed_value_exits_non_zero_with_the_error(self, capsys):
        assert main(["shell", "server:ci:abc"]) == 1
        assert "not a valid port" in capsys.readouterr().err

    def test_server_env_exports_the_shared_defaults(self, capsys):
        assert main(["server-env"]) == 0
        output = capsys.readouterr().out
        for key, value in SERVER_MODE_ENV_DEFAULTS.items():
            assert f"export {key}={value}" in output


class TestApplyServerModeEnvDefaults:
    def test_fills_missing_values_without_overriding_existing_ones(self):
        environ = {"OGX_METRICS_ENDPOINT_ENABLED": "0"}

        apply_server_mode_env_defaults(environ)

        assert environ["OGX_METRICS_ENDPOINT_ENABLED"] == "0"
        assert environ["OGX_TEST_COLLECTOR_PORT"] == SERVER_MODE_ENV_DEFAULTS["OGX_TEST_COLLECTOR_PORT"]


def _session(stack_config: str | None) -> SimpleNamespace:
    options = {"--stack-config": stack_config}
    config = SimpleNamespace(getoption=lambda name, default=None: options.get(name, default))
    return SimpleNamespace(config=config)


class TestFixtureServerStartup:
    def test_spawns_ogx_run_insecure_on_the_requested_port(self):
        with patch.object(common.subprocess, "Popen") as popen:
            common.start_ogx_server("ci-tests", 9000, log_stderr=False)

        assert popen.call_args.args[0] == ["ogx", "run", "ci-tests", "--insecure", "--port", "9000"]

    def test_server_port_suffix_is_used_for_both_the_server_and_the_client(self, monkeypatch):
        monkeypatch.delenv("OGX_CONFIG", raising=False)
        server_process = MagicMock()
        with (
            patch.object(common, "is_port_available", return_value=True),
            patch.object(common, "start_ogx_server", return_value=server_process) as start,
            patch.object(common, "wait_for_server_ready", return_value=True),
            patch.object(common, "apply_server_mode_env_defaults"),
        ):
            client = common.instantiate_ogx_client(_session("server:ci-tests:9123"))

        start.assert_called_once_with("ci-tests", 9123)
        assert str(client.base_url).startswith("http://localhost:9123")

    def test_a_server_that_exits_immediately_fails_fast_naming_the_runner_script(self, monkeypatch):
        """A child that dies on startup must fail in seconds, not after the 120s health timeout."""
        monkeypatch.delenv("OGX_CONFIG", raising=False)
        dead_process = MagicMock()
        dead_process.poll.return_value = 1
        dead_process.returncode = 1
        dead_process.stderr = None
        with (
            patch.object(common, "is_port_available", return_value=True),
            patch.object(common, "start_ogx_server", return_value=dead_process),
            patch.object(common, "apply_server_mode_env_defaults"),
            patch.object(common.requests, "get", side_effect=common.requests.exceptions.ConnectionError),
        ):
            start = time.monotonic()
            with pytest.raises(RuntimeError, match="scripts/integration-tests.sh") as exc_info:
                common.instantiate_ogx_client(_session("server:ci-tests:9124"))
            elapsed = time.monotonic() - start

        assert elapsed < 5
        assert "exited with code 1" in str(exc_info.value)

    def test_docker_is_rejected_under_bare_pytest_with_a_clear_message(self, monkeypatch):
        monkeypatch.delenv("OGX_CONFIG", raising=False)

        with pytest.raises(pytest.UsageError, match="only supported through scripts/integration-tests.sh"):
            common.instantiate_ogx_client(_session("docker:ci-tests"))

    def test_ogx_config_alone_selects_server_mode(self, monkeypatch):
        monkeypatch.setenv("OGX_CONFIG", "http://localhost:8321")

        client = common.instantiate_ogx_client(_session(None))

        assert str(client.base_url).startswith("http://localhost:8321")

    def test_missing_config_fails_with_a_clear_error(self, monkeypatch):
        monkeypatch.delenv("OGX_CONFIG", raising=False)

        with pytest.raises(ValueError, match="--stack-config or OGX_CONFIG"):
            common.instantiate_ogx_client(_session(None))
