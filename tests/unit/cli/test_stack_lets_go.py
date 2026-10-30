# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for `ogx go` and `ogx stack go` CLI commands."""

import argparse
import base64
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import venv
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from ogx.cli.letsgo import LetsGo, LetsGoDeprecated
from ogx.cli.stack.lets_go import (
    _CLAUDE_CODE_ALIASES,
    _CLAUDE_CODE_PROVIDER_PRIORITY,
    StackLetsGo,
    _autodetect_providers,
    _build_claude_code_aliases,
    _install_provider_deps,
    _is_reachable,
    _pip_install_command,
    _ProbeStatus,
)
from ogx.core.build import split_pip_packages


@pytest.fixture
def lets_go() -> StackLetsGo:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return StackLetsGo(subparsers)


@pytest.fixture
def top_level_letsgo() -> LetsGo:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return LetsGo(subparsers)


@pytest.fixture
def top_level_letsgo_deprecated() -> LetsGoDeprecated:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return LetsGoDeprecated(subparsers)


class TestArguments:
    def test_defaults(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args([])
        assert args.port == 8321
        assert args.enable_ui is False
        assert args.persist_config is False
        assert args.providers_override is None

    def test_port_override(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args(["--port", "9000"])
        assert args.port == 9000

    def test_enable_ui_flag(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args(["--enable-ui"])
        assert args.enable_ui is True

    def test_persist_config_flag(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args(["--persist-config"])
        assert args.persist_config is True

    def test_providers_override_flag(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args(["--providers-override", "inference=remote::ollama"])
        assert args.providers_override == "inference=remote::ollama"

    def test_skip_install_deps_default(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args([])
        assert args.skip_install_deps is False

    def test_skip_install_deps_flag(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args(["--skip-install-deps"])
        assert args.skip_install_deps is True

    def test_port_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OGX_PORT", "9999")
        subparsers = argparse.ArgumentParser().add_subparsers()
        instance = StackLetsGo(subparsers)
        args = instance.parser.parse_args([])
        assert args.port == 9999


class TestTopLevelLetsGoArguments:
    def test_defaults(self, top_level_letsgo: LetsGo):
        args = top_level_letsgo.parser.parse_args([])
        assert args.port == 8321
        assert args.enable_ui is False
        assert args.persist_config is False
        assert args.providers_override is None

    def test_all_options(self, top_level_letsgo: LetsGo):
        args = top_level_letsgo.parser.parse_args(
            [
                "--port",
                "9000",
                "--enable-ui",
                "--persist-config",
                "--providers-override",
                "inference=remote::ollama",
                "--skip-install-deps",
            ]
        )
        assert args.port == 9000
        assert args.enable_ui is True
        assert args.persist_config is True
        assert args.providers_override == "inference=remote::ollama"
        assert args.skip_install_deps is True


class TestAutodetect:
    async def test_autodetect_no_providers(self):
        with patch("ogx.cli.stack.lets_go._probe_provider_availability") as m:
            m.return_value = (_ProbeStatus.UNREACHABLE, [], "", "default", None)
            parts = (await _autodetect_providers())[0].split(",")
            assert "files=inline::localfs" in parts
            assert "vector_io=inline::faiss" in parts

    async def test_no_key_providers_excluded(self):
        with patch("ogx.cli.stack.lets_go._probe_provider_availability") as m:
            m.return_value = (_ProbeStatus.NO_KEY, [], "", "default", None)
            parts = (await _autodetect_providers())[0].split(",")
            assert "files=inline::localfs" in parts
            assert "vector_io=inline::faiss" in parts

    async def test_autodetect_all_ok(self):
        with patch("ogx.cli.stack.lets_go._probe_provider_availability") as m:
            m.return_value = (_ProbeStatus.OK, [], "http://test", "default", None)
            spec, _ = await _autodetect_providers()
            parts = spec.split(",")
            assert "inference=remote::ollama" in parts
            assert "inference=remote::anthropic" in parts
            assert "files=inline::localfs" in parts

    async def test_autodetect_only_ollama(self):
        def side_effect(
            provider_type: str,
            base_url_env: object,
            default_base_url: str,
            required_api_key_env: object,
            optional_api_key_env: object = None,
            debug: bool = False,
        ) -> tuple:
            if provider_type == "remote::ollama":
                return (_ProbeStatus.OK, [], "http://localhost:11434/v1", "default", None)
            return (_ProbeStatus.UNREACHABLE, [], "", "default", None)

        with patch("ogx.cli.stack.lets_go._probe_provider_availability", side_effect=side_effect):
            parts = (await _autodetect_providers())[0].split(",")
            assert "inference=remote::ollama" in parts
            assert len(parts) == 6  # 1 inference + 5 inline

    async def test_autodetect_uses_env_var_name(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OLLAMA_URL", "http://myhost:11434/v1")
        captured: list[str] = []

        def side_effect(
            provider_type: str,
            base_url_env: object,
            default_base_url: str,
            required_api_key_env: object,
            optional_api_key_env: object = None,
            debug: bool = False,
        ) -> tuple:
            if provider_type == "remote::ollama":
                captured.append(base_url_env)
            return (_ProbeStatus.UNREACHABLE, [], "", "default", None)

        with patch("ogx.cli.stack.lets_go._probe_provider_availability", side_effect=side_effect):
            await _autodetect_providers()
            assert captured[0] == "OLLAMA_URL"

    async def test_autodetect_typesafe_candidate_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
        captured: list[tuple] = []

        def side_effect(
            provider_type: str,
            base_url_env: object,
            default_base_url: str,
            required_api_key_env: object,
            optional_api_key_env: object = None,
            debug: bool = False,
        ) -> tuple:
            if provider_type == "remote::typesafe":
                captured.append((base_url_env, default_base_url, required_api_key_env))
                return (_ProbeStatus.OK, [], "https://api.typesafe.ai", "default", None)
            return (_ProbeStatus.UNREACHABLE, [], "", "default", None)

        with patch("ogx.cli.stack.lets_go._probe_provider_availability", side_effect=side_effect):
            parts = (await _autodetect_providers())[0].split(",")
            assert "inference=remote::typesafe" in parts
            assert captured == [("TYPESAFE_BASE_URL", "https://api.typesafe.ai", "TYPESAFE_API_KEY")]

    async def test_autodetect_result_order_matches_candidate_order(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        def side_effect(
            provider_type: str,
            base_url_env: object,
            default_base_url: str,
            required_api_key_env: object,
            optional_api_key_env: object = None,
            debug: bool = False,
        ) -> tuple:
            if provider_type in ("remote::ollama", "remote::openai"):
                return (_ProbeStatus.OK, [], "http://test", "default", None)
            return (_ProbeStatus.UNREACHABLE, [], "", "default", None)

        with patch("ogx.cli.stack.lets_go._probe_provider_availability", side_effect=side_effect):
            parts = (await _autodetect_providers())[0].split(",")
            assert parts.index("inference=remote::ollama") < parts.index("inference=remote::openai")

    async def test_autodetect_includes_vllm_on_needs_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("VLLM_API_TOKEN", raising=False)

        def side_effect(
            provider_type: str,
            base_url_env: object,
            default_base_url: str,
            required_api_key_env: object,
            optional_api_key_env: object = None,
            debug: bool = False,
        ) -> tuple:
            if provider_type == "remote::vllm":
                return (_ProbeStatus.NEEDS_KEY, [], "http://localhost:8000/v1", "default", None)
            return (_ProbeStatus.UNREACHABLE, [], "", "default", None)

        with patch("ogx.cli.stack.lets_go._probe_provider_availability", side_effect=side_effect):
            parts = (await _autodetect_providers())[0].split(",")
            assert "inference=remote::vllm" in parts

    async def test_autodetect_detects_embedding_when_no_key_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        def side_effect(
            provider_type: str,
            base_url_env: object,
            default_base_url: str,
            required_api_key_env: object,
            optional_api_key_env: object = None,
            debug: bool = False,
        ) -> tuple:
            if provider_type == "remote::openai":
                # No key → NO_KEY status
                return (_ProbeStatus.NO_KEY, [], "https://api.openai.com/v1", "default", None)
            return (_ProbeStatus.UNREACHABLE, [], "", "default", None)

        with patch("ogx.cli.stack.lets_go._probe_provider_availability", side_effect=side_effect):
            spec, embedding = await _autodetect_providers()
            assert embedding is None


_OLLAMA_MISSING_DEPS = (
    _ProbeStatus.MISSING_DEPS,
    [],
    "http://localhost:11434/v1",
    "default",
    ["typesafe-sdk>=0.7.2", "torch --index-url https://download.pytorch.org/whl/cpu"],
)
_OLLAMA_OK = (_ProbeStatus.OK, [], "http://localhost:11434/v1", "default", None)
_UNREACHABLE = (_ProbeStatus.UNREACHABLE, [], "", "default", None)


_STUB_PACKAGE = "ogx_install_target_stub"


def _build_stub_wheel(wheels_dir: Path, name: str) -> None:
    """Write a minimal pure-python wheel for `name` 0.0.1, with a valid RECORD, into wheels_dir."""
    dist_info = f"{name}-0.0.1.dist-info"
    files = {
        f"{name}/__init__.py": b"",
        f"{dist_info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.0.1\n".encode(),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: ogx-tests\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record_lines = []
    for path, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        record_lines.append(f"{path},sha256={digest},{len(content)}")
    record_lines.append(f"{dist_info}/RECORD,,")
    files[f"{dist_info}/RECORD"] = ("\n".join(record_lines) + "\n").encode()

    with zipfile.ZipFile(wheels_dir / f"{name}-0.0.1-py3-none-any.whl", "w") as wheel:
        for path, content in files.items():
            wheel.writestr(path, content)


def _ollama_probe(*ollama_results: tuple) -> Callable[..., tuple]:
    """Return a probe side effect that yields ollama_results in order and UNREACHABLE for others."""
    remaining = list(ollama_results)

    def side_effect(provider_type: str, *args: object, **kwargs: object) -> tuple:
        if provider_type == "remote::ollama":
            return remaining.pop(0)
        return _UNREACHABLE

    return side_effect


class TestInstallMissingDeps:
    async def test_installs_registry_packages_and_reprobes_reachable_provider(self):
        probe = AsyncMock(side_effect=_ollama_probe(_OLLAMA_MISSING_DEPS, _OLLAMA_OK))
        with (
            patch("ogx.cli.stack.lets_go._probe_provider_availability", probe),
            patch("ogx.cli.stack.lets_go._is_reachable", AsyncMock(return_value=True)),
            patch("ogx.cli.stack.lets_go._install_provider_deps", return_value=True) as mock_install,
        ):
            parts = (await _autodetect_providers(install_missing_deps=True))[0].split(",")

        assert "inference=remote::ollama" in parts
        mock_install.assert_called_once_with(
            ["typesafe-sdk>=0.7.2"], ["torch --index-url https://download.pytorch.org/whl/cpu"]
        )
        ollama_calls = [c for c in probe.call_args_list if c.args[0] == "remote::ollama"]
        assert len(ollama_calls) == 2

    async def test_skips_install_when_provider_is_unreachable(self):
        with (
            patch(
                "ogx.cli.stack.lets_go._probe_provider_availability",
                side_effect=_ollama_probe(_OLLAMA_MISSING_DEPS),
            ),
            patch("ogx.cli.stack.lets_go._is_reachable", AsyncMock(return_value=False)),
            patch("ogx.cli.stack.lets_go._install_provider_deps") as mock_install,
        ):
            parts = (await _autodetect_providers(install_missing_deps=True))[0].split(",")

        assert "inference=remote::ollama" not in parts
        mock_install.assert_not_called()

    async def test_skips_install_by_default(self):
        mock_reachable = AsyncMock(return_value=True)
        with (
            patch(
                "ogx.cli.stack.lets_go._probe_provider_availability",
                side_effect=_ollama_probe(_OLLAMA_MISSING_DEPS),
            ),
            patch("ogx.cli.stack.lets_go._is_reachable", mock_reachable),
            patch("ogx.cli.stack.lets_go._install_provider_deps") as mock_install,
        ):
            parts = (await _autodetect_providers())[0].split(",")

        assert "inference=remote::ollama" not in parts
        mock_reachable.assert_not_called()
        mock_install.assert_not_called()

    async def test_does_not_reprobe_when_install_fails(self):
        probe = AsyncMock(side_effect=_ollama_probe(_OLLAMA_MISSING_DEPS))
        with (
            patch("ogx.cli.stack.lets_go._probe_provider_availability", probe),
            patch("ogx.cli.stack.lets_go._is_reachable", AsyncMock(return_value=True)),
            patch("ogx.cli.stack.lets_go._install_provider_deps", return_value=False),
        ):
            parts = (await _autodetect_providers(install_missing_deps=True))[0].split(",")

        assert "inference=remote::ollama" not in parts
        ollama_calls = [c for c in probe.call_args_list if c.args[0] == "remote::ollama"]
        assert len(ollama_calls) == 1

    def test_pip_install_targets_running_interpreter_with_uv(self):
        with patch("ogx.cli.stack.lets_go.shutil.which", return_value="/usr/bin/uv"):
            assert _pip_install_command() == ["uv", "pip", "install", "--python", sys.executable]

    def test_pip_install_falls_back_to_pip_without_uv(self):
        with patch("ogx.cli.stack.lets_go.shutil.which", return_value=None):
            assert _pip_install_command() == [sys.executable, "-m", "pip", "install"]

    def test_install_provider_deps_summarizes_failure(self, capsys: pytest.CaptureFixture[str]):
        pep_668_error = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr=(
                "error: externally-managed-environment\n\n"
                "This environment is externally managed\n"
                "To install Python packages system-wide, try apt install python3-xyz.\n"
                + "\n".join(f"    hint line {i}" for i in range(20))
            ),
        )
        with patch("ogx.cli.stack.lets_go.subprocess.run", return_value=pep_668_error):
            assert _install_provider_deps(["typesafe-sdk>=0.7.2"], []) is False

        err = capsys.readouterr().err
        assert "Failed to install typesafe-sdk>=0.7.2" in err
        assert "error: externally-managed-environment" in err
        assert "PEP 668" in err
        assert "hint line" not in err

    def test_install_provider_deps_reports_success(self):
        ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("ogx.cli.stack.lets_go.subprocess.run", return_value=ok) as mock_run:
            assert _install_provider_deps(["typesafe-sdk>=0.7.2"], ["torch --index-url https://example"]) is True
        assert mock_run.call_count == 2

    def test_install_provider_deps_targets_the_running_interpreter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Packages land in the interpreter running `ogx go` (sys.executable), not the caller's environment.

        Under `uvx ogx go` these differ: without `--python`, `uv pip install` would target
        $VIRTUAL_ENV or a .venv found from the working directory instead. Hermetic: the venv
        is created offline and the package comes from a local wheel with --no-index.
        """
        monkeypatch.chdir(tmp_path)
        venv_dir = tmp_path / "venv"
        # uv installs into a venv without pip; the pip fallback needs pip, from ensurepip's bundled wheel.
        venv.create(venv_dir, with_pip=shutil.which("uv") is None)
        venv_python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        wheels_dir = tmp_path / "wheels"
        wheels_dir.mkdir()
        _build_stub_wheel(wheels_dir, _STUB_PACKAGE)

        normal, special = split_pip_packages([f"{_STUB_PACKAGE} --no-index --find-links {wheels_dir}"])
        with patch("ogx.cli.stack.lets_go.sys.executable", str(venv_python)):
            assert _install_provider_deps(normal, special) is True

        subprocess.run([str(venv_python), "-c", f"import {_STUB_PACKAGE}"], check=True)
        assert importlib.util.find_spec(_STUB_PACKAGE) is None

    async def test_is_reachable_without_base_url(self):
        assert await _is_reachable("") is False

    async def test_is_reachable_on_connection_error(self):
        with patch("ogx.cli.stack.lets_go.httpx2.AsyncClient.get", side_effect=httpx2.ConnectError("refused")):
            assert await _is_reachable("http://localhost:11434/v1") is False

    async def test_is_reachable_on_any_http_response(self):
        with patch("ogx.cli.stack.lets_go.httpx2.AsyncClient.get", AsyncMock(return_value=MagicMock(status_code=404))):
            assert await _is_reachable("http://localhost:11434/v1") is True

    async def test_is_reachable_skips_tls_verification(self):
        client = MagicMock()
        client.get = AsyncMock(return_value=MagicMock(status_code=200))
        client_cls = MagicMock()
        client_cls.return_value.__aenter__ = AsyncMock(return_value=client)
        client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("ogx.cli.stack.lets_go.httpx2.AsyncClient", client_cls):
            assert await _is_reachable("https://localhost:8000/v1") is True
        assert client_cls.call_args.kwargs["verify"] is False

    def test_split_pip_packages_separates_flagged_entries(self):
        normal, special = split_pip_packages(
            ["ollama", "torch --index-url https://example", "x --no-deps", "y --no-index --find-links /wheels"]
        )
        assert normal == ["ollama"]
        assert special == ["torch --index-url https://example", "x --no-deps", "y --no-index --find-links /wheels"]


class TestRunCommandSync:
    """Tests that interact with the sync CLI entry point via _run_stack_lets_go_cmd.

    These patch `_autodetect_providers` with AsyncMock because it is now async.
    run_letsgo_cmd internally calls asyncio.run(), so these tests must not use
    the async test marker (which would have a running event loop already).
    """

    def test_no_inference_provider_exits(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args([])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with pytest.raises(SystemExit):
                with patch(
                    "ogx.cli.stack.lets_go._autodetect_providers",
                    AsyncMock(
                        return_value=(
                            "files=inline::localfs,vector_io=inline::faiss,tool_runtime=inline::file-search,responses=inline::builtin",
                            None,
                        )
                    ),
                ):
                    lets_go._run_stack_lets_go_cmd(args)

    def test_empty_spec_exits(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args([])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with pytest.raises(SystemExit):
                with patch("ogx.cli.stack.lets_go._autodetect_providers", AsyncMock(return_value=("", None))):
                    lets_go._run_stack_lets_go_cmd(args)

    def test_run_command_uses_autodetected_providers(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args([])
        mock_cfg = MagicMock()
        mock_cfg.model_dump.return_value = {}

        with (
            patch("ogx.cli.stack.lets_go._uvicorn_run"),
            patch("ogx.cli.stack.lets_go.get_provider_dependencies", return_value=([], [], [])),
            patch("ogx.cli.stack.lets_go.subprocess.run"),
            patch("ogx.cli.stack.lets_go.run_config_from_dynamic_config_spec", return_value=mock_cfg),
            patch(
                "ogx.cli.stack.lets_go._autodetect_providers",
                AsyncMock(return_value=("inference=remote::ollama", None)),
            ),
            patch("builtins.open", MagicMock()),
            patch("ogx.cli.stack.lets_go.yaml.dump"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            lets_go._run_stack_lets_go_cmd(args)

        assert mock_cfg.model_dump.called

    @pytest.mark.parametrize(("cli_args", "expected"), [([], True), (["--skip-install-deps"], False)])
    def test_autodetect_installs_missing_deps_unless_skipped(
        self, lets_go: StackLetsGo, cli_args: list[str], expected: bool
    ):
        args = lets_go.parser.parse_args(cli_args)
        mock_cfg = MagicMock()
        mock_cfg.model_dump.return_value = {}
        mock_detect = AsyncMock(return_value=("inference=remote::ollama", None))

        with (
            patch("ogx.cli.stack.lets_go._uvicorn_run"),
            patch("ogx.cli.stack.lets_go.get_provider_dependencies", return_value=([], [], [])),
            patch("ogx.cli.stack.lets_go.subprocess.run"),
            patch("ogx.cli.stack.lets_go.run_config_from_dynamic_config_spec", return_value=mock_cfg),
            patch("ogx.cli.stack.lets_go._autodetect_providers", mock_detect),
            patch("builtins.open", MagicMock()),
            patch("ogx.cli.stack.lets_go.yaml.dump"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            lets_go._run_stack_lets_go_cmd(args)

        assert mock_detect.call_args.kwargs["install_missing_deps"] is expected

    def test_providers_override_skips_autodetect(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args(["--providers-override", "inference=remote::ollama"])
        mock_cfg = MagicMock()
        mock_cfg.model_dump.return_value = {}

        with (
            patch("ogx.cli.stack.lets_go._uvicorn_run"),
            patch("ogx.cli.stack.lets_go.get_provider_dependencies", return_value=([], [], [])),
            patch("ogx.cli.stack.lets_go.subprocess.run"),
            patch("ogx.cli.stack.lets_go.run_config_from_dynamic_config_spec", return_value=mock_cfg),
            patch("ogx.cli.stack.lets_go._autodetect_providers") as mock_detect,
            patch("builtins.open", MagicMock()),
            patch("ogx.cli.stack.lets_go.yaml.dump"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            lets_go._run_stack_lets_go_cmd(args)
        mock_detect.assert_not_called()


class TestDeprecation:
    def test_stack_letsgo_emits_deprecation_warning(self, lets_go: StackLetsGo):
        with (
            patch("ogx.cli.stack.lets_go.run_letsgo_cmd"),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            args = lets_go.parser.parse_args([])
            lets_go._run_stack_lets_go_cmd(args)

        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 1
        assert "deprecated" in str(future_warnings[0].message)
        assert "ogx go" in str(future_warnings[0].message)

    def test_top_level_letsgo_no_deprecation_warning(self, top_level_letsgo: LetsGo):
        with (
            patch("ogx.cli.letsgo.run_letsgo_cmd"),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            args = top_level_letsgo.parser.parse_args([])
            top_level_letsgo._run_cmd(args)

        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 0

    def test_top_level_letsgo_deprecated_emits_deprecation_warning(self, top_level_letsgo_deprecated: LetsGoDeprecated):
        with (
            patch("ogx.cli.letsgo.run_letsgo_cmd"),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            args = top_level_letsgo_deprecated.parser.parse_args([])
            top_level_letsgo_deprecated._run_cmd(args)

        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 1
        assert "deprecated" in str(future_warnings[0].message)
        assert "ogx go" in str(future_warnings[0].message)


class TestClaudeCodeAliases:
    def test_anthropic_chosen_over_others(self):
        spec = "inference=remote::anthropic,inference=remote::ollama,files=inline::localfs"
        aliases = _build_claude_code_aliases(spec)
        assert len(aliases) == len(_CLAUDE_CODE_ALIASES)
        assert all(a.provider_id == "anthropic" for a in aliases)

    def test_anthropic_uses_direct_model_id(self):
        spec = "inference=remote::anthropic"
        aliases = _build_claude_code_aliases(spec)
        for alias in aliases:
            assert alias.provider_model_id == alias.model_id

    def test_ollama_fallback_uses_auto(self):
        spec = "inference=remote::ollama,files=inline::localfs"
        aliases = _build_claude_code_aliases(spec)
        assert all(a.provider_id == "ollama" for a in aliases)
        assert all(a.provider_model_id == "auto" for a in aliases)

    def test_vllm_fallback_uses_auto(self):
        spec = "inference=remote::vllm"
        aliases = _build_claude_code_aliases(spec)
        assert all(a.provider_id == "vllm" for a in aliases)
        assert all(a.provider_model_id == "auto" for a in aliases)

    def test_openai_fallback_uses_auto(self):
        spec = "inference=remote::openai"
        aliases = _build_claude_code_aliases(spec)
        assert all(a.provider_id == "openai" for a in aliases)
        assert all(a.provider_model_id == "auto" for a in aliases)

    def test_priority_order_ollama_before_openai(self):
        spec = "inference=remote::openai,inference=remote::ollama"
        aliases = _build_claude_code_aliases(spec)
        assert all(a.provider_id == "ollama" for a in aliases)

    def test_unknown_provider_returns_empty(self):
        spec = "inference=remote::llama-openai-compat"
        aliases = _build_claude_code_aliases(spec)
        assert aliases == []

    def test_no_inference_returns_empty(self):
        spec = "files=inline::localfs,vector_io=inline::faiss"
        aliases = _build_claude_code_aliases(spec)
        assert aliases == []

    def test_all_aliases_present(self):
        spec = "inference=remote::anthropic"
        aliases = _build_claude_code_aliases(spec)
        alias_model_ids = [a.model_id for a in aliases]
        for expected in _CLAUDE_CODE_ALIASES:
            assert expected in alias_model_ids

    @pytest.mark.parametrize(
        "model_id",
        ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5", "claude-fable-5", "claude-mythos-5"],
    )
    def test_current_generation_models_are_registered_for_anthropic(self, model_id):
        """The claude CLI requests these unprefixed IDs, and an unregistered one fails to resolve."""
        aliases = _build_claude_code_aliases("inference=remote::anthropic")

        alias = next(a for a in aliases if a.model_id == model_id)
        assert alias.provider_id == "anthropic"
        assert alias.provider_model_id == model_id

    def test_previous_generation_models_stay_registered_for_older_cli_versions(self):
        assert {"claude-sonnet-4-6", "claude-opus-4-7"} <= set(_CLAUDE_CODE_ALIASES)

    def test_alias_ids_are_unique(self):
        assert len(_CLAUDE_CODE_ALIASES) == len(set(_CLAUDE_CODE_ALIASES))

    def test_newest_model_of_each_tier_comes_first(self):
        """`ogx connect claude` maps each tier to the first matching model, so newer ones must precede older."""
        assert _CLAUDE_CODE_ALIASES.index("claude-sonnet-5") < _CLAUDE_CODE_ALIASES.index("claude-sonnet-4-6")
        assert _CLAUDE_CODE_ALIASES.index("claude-opus-5") < _CLAUDE_CODE_ALIASES.index("claude-opus-4-7")

    def test_connect_claude_picks_the_newest_model_per_tier(self):
        from ogx.cli.connect.claude import _detect_tier_models

        assert _detect_tier_models(list(_CLAUDE_CODE_ALIASES)) == {
            "haiku": "claude-haiku-4-5",
            "sonnet": "claude-sonnet-5",
            "opus": "claude-opus-5",
        }

    def test_aliases_have_unprefixed_metadata(self):
        spec = "inference=remote::anthropic"
        aliases = _build_claude_code_aliases(spec)
        for alias in aliases:
            assert alias.metadata is not None
            assert alias.metadata.get("_unprefixed_alias") is True

    def test_priority_list_covers_expected_providers(self):
        assert "anthropic" in _CLAUDE_CODE_PROVIDER_PRIORITY
        assert "ollama" in _CLAUDE_CODE_PROVIDER_PRIORITY


class TestAddFileSearchAndResponses:
    def test_uses_brave_when_brave_key_is_set(self, monkeypatch: pytest.MonkeyPatch):
        from ogx.cli.stack.lets_go import _add_file_search_and_responses
        from ogx.core.datatypes import StackConfig

        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")
        monkeypatch.delenv("TAVILY_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("BING_API_KEY", raising=False)
        monkeypatch.delenv("NIMBLE_API_KEY", raising=False)

        with patch("ogx.cli.stack.lets_go.cprint") as mock_cprint:
            _add_file_search_and_responses(
                StackConfig(
                    distro_name="test",
                    providers={
                        "tool_runtime": [],
                        "responses": [],
                    },
                )
            )

        web_search_providers = [
            p
            for p in ["brave-search", "tavily-search", "bing-search", "nimble-search"]
            if any(p in str(call) for call in mock_cprint.call_args_list)
        ]
        # Only brave should be added (env var set)
        assert "brave-search" in web_search_providers
        assert "tavily-search" not in web_search_providers
        assert "bing-search" not in web_search_providers
        assert "nimble-search" not in web_search_providers

    def test_falls_back_to_tavily_when_only_tavily_key_set(self, monkeypatch: pytest.MonkeyPatch):
        from ogx.cli.stack.lets_go import _add_file_search_and_responses
        from ogx.core.datatypes import StackConfig

        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("TAVILY_SEARCH_API_KEY", "tavily-key")
        monkeypatch.delenv("BING_API_KEY", raising=False)
        monkeypatch.delenv("NIMBLE_API_KEY", raising=False)

        with patch("ogx.cli.stack.lets_go.cprint") as mock_cprint:
            _add_file_search_and_responses(
                StackConfig(
                    distro_name="test",
                    providers={
                        "tool_runtime": [],
                        "responses": [],
                    },
                )
            )

        provider_ids_in_calls = set()
        for call in mock_cprint.call_args_list:
            if "brave-search" in str(call):
                provider_ids_in_calls.add("brave-search")
            if "tavily-search" in str(call):
                provider_ids_in_calls.add("tavily-search")
            if "bing-search" in str(call):
                provider_ids_in_calls.add("bing-search")
            if "nimble-search" in str(call):
                provider_ids_in_calls.add("nimble-search")

        # Only tavily is added since only its env var is set
        assert provider_ids_in_calls == {"tavily-search"}

    def test_selects_first_with_api_key(self, monkeypatch: pytest.MonkeyPatch):
        from ogx.cli.stack.lets_go import _add_file_search_and_responses
        from ogx.core.datatypes import StackConfig

        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")
        monkeypatch.setenv("TAVILY_SEARCH_API_KEY", "tavily-key")
        monkeypatch.setenv("BING_API_KEY", "bing-key")
        monkeypatch.setenv("NIMBLE_API_KEY", "nimble-key")

        with patch("ogx.cli.stack.lets_go.cprint") as mock_cprint:
            _add_file_search_and_responses(
                StackConfig(
                    distro_name="test",
                    providers={
                        "tool_runtime": [],
                        "responses": [],
                    },
                )
            )

        provider_ids_in_calls = set()
        for call in mock_cprint.call_args_list:
            if "brave-search" in str(call):
                provider_ids_in_calls.add("brave-search")
            if "tavily-search" in str(call):
                provider_ids_in_calls.add("tavily-search")
            if "bing-search" in str(call):
                provider_ids_in_calls.add("bing-search")
            if "nimble-search" in str(call):
                provider_ids_in_calls.add("nimble-search")

        # All four have keys, all four should be added
        assert provider_ids_in_calls == {"brave-search", "tavily-search", "bing-search", "nimble-search"}

    def test_no_web_search_when_no_key_set(self, monkeypatch: pytest.MonkeyPatch):
        from ogx.cli.stack.lets_go import _add_file_search_and_responses
        from ogx.core.datatypes import StackConfig

        # Clear env vars to ensure no keys are set
        for var in ("BRAVE_SEARCH_API_KEY", "TAVILY_SEARCH_API_KEY", "BING_API_KEY", "NIMBLE_API_KEY", "EXA_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        with patch("ogx.cli.stack.lets_go.cprint") as mock_cprint:
            _add_file_search_and_responses(
                StackConfig(
                    distro_name="test",
                    providers={
                        "tool_runtime": [],
                        "responses": [],
                    },
                )
            )

        # Should show disabled message, not provider names
        has_disabled = any("web search disabled" in str(call) for call in mock_cprint.call_args_list)
        assert has_disabled

    def test_duplicate_providers_when_already_configured(self, monkeypatch: pytest.MonkeyPatch):
        from ogx.cli.stack.lets_go import _add_file_search_and_responses
        from ogx.core.datatypes import Provider, StackConfig

        for var in ("BRAVE_SEARCH_API_KEY", "TAVILY_SEARCH_API_KEY", "BING_API_KEY", "NIMBLE_API_KEY", "EXA_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        initial_providers = [
            Provider(provider_id="brave-search", provider_type="remote::brave-search"),
        ]

        config = StackConfig(
            distro_name="test",
            providers={
                "tool_runtime": initial_providers.copy(),
                "responses": [],
            },
        )

        with patch("ogx.cli.stack.lets_go.cprint"):
            _add_file_search_and_responses(config)

        web_search_types = {
            "remote::brave-search",
            "remote::tavily-search",
            "remote::bing-search",
            "remote::nimble-search",
        }
        web_search_providers = [p for p in config.providers["tool_runtime"] if p.provider_type in web_search_types]
        assert len(web_search_providers) == 1

    def test_existing_provider_keeps_no_api_key_config(self):
        from ogx.cli.stack.lets_go import _add_file_search_and_responses
        from ogx.core.datatypes import Provider, StackConfig

        config = StackConfig(
            distro_name="test",
            providers={
                "tool_runtime": [
                    Provider(
                        provider_id="brave-search",
                        provider_type="remote::brave-search",
                        config={"api_key": "existing-key", "max_results": 5},
                    ),
                ],
                "responses": [],
            },
        )

        with patch("ogx.cli.stack.lets_go.cprint"):
            _add_file_search_and_responses(config)

        web_search_providers = [
            p for p in config.providers["tool_runtime"] if p.provider_type == "remote::brave-search"
        ]
        assert len(web_search_providers) == 1
        assert web_search_providers[0].config["api_key"] == "existing-key"
        assert web_search_providers[0].config["max_results"] == 5
        assert _CLAUDE_CODE_PROVIDER_PRIORITY.index("anthropic") < _CLAUDE_CODE_PROVIDER_PRIORITY.index("ollama")
