# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for `ogx letsgo` and `ogx stack letsgo` CLI commands."""

import argparse
import warnings
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ogx.cli.letsgo import LetsGo
from ogx.cli.stack.lets_go import (
    _CLAUDE_CODE_ALIASES,
    _CLAUDE_CODE_PROVIDER_PRIORITY,
    StackLetsGo,
    _build_claude_code_aliases,
    _probe_endpoint,
    _ProbeStatus,
)


@pytest.fixture
def lets_go() -> StackLetsGo:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return StackLetsGo(subparsers)


@pytest.fixture
def top_level_letsgo() -> LetsGo:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return LetsGo(subparsers)


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


class TestProbeEndpoint:
    def test_empty_base_url_returns_unreachable(self):
        assert _probe_endpoint("", "models", False, None) == _ProbeStatus.UNREACHABLE

    def test_url_construction_preserves_v1_path(self):
        """Without a trailing slash urljoin strips the last path segment."""
        captured: list[str] = []

        def fake_get(url: str, **kwargs: object) -> None:
            captured.append(url)
            raise OSError("offline")

        with patch("ogx.cli.stack.lets_go.httpx.get", side_effect=fake_get):
            _probe_endpoint("http://localhost:11434/v1", "models", False, None)

        assert captured[0] == "http://localhost:11434/v1/models"

    def test_ok_on_200(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        with patch("ogx.cli.stack.lets_go.httpx.get", return_value=mock_resp):
            assert _probe_endpoint("http://localhost:11434/v1", "models", False, None) == _ProbeStatus.OK

    def test_ok_on_204(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 204
        with patch("ogx.cli.stack.lets_go.httpx.get", return_value=mock_resp):
            assert _probe_endpoint("http://localhost:8000/v1", "health", False, None) == _ProbeStatus.OK

    def test_no_key_when_env_var_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = _probe_endpoint("https://api.openai.com/v1", "models", True, "OPENAI_API_KEY")
        assert result == _ProbeStatus.NO_KEY

    def test_no_key_when_key_env_is_none(self):
        result = _probe_endpoint("https://example.com/v1", "models", True, None)
        assert result == _ProbeStatus.NO_KEY

    def test_auth_on_401(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        with patch("ogx.cli.stack.lets_go.httpx.get", return_value=mock_resp):
            assert _probe_endpoint("https://api.openai.com/v1", "models", True, "OPENAI_API_KEY") == _ProbeStatus.AUTH

    def test_auth_on_403(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 403
        with patch("ogx.cli.stack.lets_go.httpx.get", return_value=mock_resp):
            assert _probe_endpoint("https://api.openai.com/v1", "models", True, "OPENAI_API_KEY") == _ProbeStatus.AUTH

    def test_unreachable_on_400(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        with patch("ogx.cli.stack.lets_go.httpx.get", return_value=mock_resp):
            assert _probe_endpoint("http://localhost:11434/v1", "models", False, None) == _ProbeStatus.UNREACHABLE

    def test_unreachable_on_500(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 500
        with patch("ogx.cli.stack.lets_go.httpx.get", return_value=mock_resp):
            assert _probe_endpoint("http://localhost:11434/v1", "models", False, None) == _ProbeStatus.UNREACHABLE

    def test_unreachable_on_connection_error(self):
        with patch("ogx.cli.stack.lets_go.httpx.get", side_effect=OSError("connection refused")):
            assert _probe_endpoint("http://localhost:11434/v1", "models", False, None) == _ProbeStatus.UNREACHABLE

    def test_unreachable_on_timeout(self):
        with patch("ogx.cli.stack.lets_go.httpx.get", side_effect=httpx.TimeoutException("timeout")):
            assert _probe_endpoint("http://localhost:11434/v1", "models", False, None) == _ProbeStatus.UNREACHABLE

    def test_extra_headers_forwarded(self):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        with patch("ogx.cli.stack.lets_go.httpx.get", return_value=mock_resp) as mock_get:
            _probe_endpoint(
                "https://api.anthropic.com/v1",
                "models",
                False,
                None,
                {"anthropic-version": "2023-06-01"},
            )
        assert mock_get.call_args.kwargs["headers"].get("anthropic-version") == "2023-06-01"

    def test_auth_headers_set_when_key_present(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        with patch("ogx.cli.stack.lets_go.httpx.get", return_value=mock_resp) as mock_get:
            _probe_endpoint("https://api.openai.com/v1", "models", True, "OPENAI_API_KEY")
        headers = mock_get.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-secret"
        assert headers["x-api-key"] == "sk-secret"


class TestAutodetect:
    @patch("ogx.cli.stack.lets_go._probe_endpoint", return_value=_ProbeStatus.UNREACHABLE)
    def test_autodetect_no_providers(self, mock_probe: MagicMock):
        from ogx.cli.stack.lets_go import _autodetect_providers

        parts = _autodetect_providers().split(",")
        assert "files=inline::localfs" in parts
        assert "vector_io=inline::faiss" in parts
        assert "tool_runtime=inline::file-search" in parts
        assert "responses=inline::builtin" in parts

    @patch("ogx.cli.stack.lets_go._probe_endpoint", return_value=_ProbeStatus.NO_KEY)
    def test_no_key_providers_excluded(self, mock_probe: MagicMock):
        from ogx.cli.stack.lets_go import _autodetect_providers

        parts = _autodetect_providers().split(",")
        assert "files=inline::localfs" in parts
        assert "vector_io=inline::faiss" in parts
        assert "tool_runtime=inline::file-search" in parts
        assert "responses=inline::builtin" in parts

    @patch("ogx.cli.stack.lets_go._probe_endpoint", return_value=_ProbeStatus.OK)
    def test_autodetect_all_ok(self, mock_probe: MagicMock):
        from ogx.cli.stack.lets_go import _autodetect_providers

        result = _autodetect_providers()
        parts = result.split(",")
        assert "inference=remote::ollama" in parts
        assert "inference=remote::anthropic" in parts
        assert "files=inline::localfs" in parts
        assert "responses=inline::builtin" in parts
        assert len(parts) == 12  # 6 probed + 6 inline

    @patch("ogx.cli.stack.lets_go._probe_endpoint")
    def test_autodetect_only_ollama(self, mock_probe: MagicMock):
        from ogx.cli.stack.lets_go import _autodetect_providers

        def side_effect(
            base_url: str, probe_path: str, requires_key: bool, key_env: object, extra_headers: object = None
        ) -> _ProbeStatus:
            if "11434" in base_url:
                return _ProbeStatus.OK
            return _ProbeStatus.UNREACHABLE

        mock_probe.side_effect = side_effect
        parts = _autodetect_providers().split(",")
        assert "inference=remote::ollama" in parts
        assert "files=inline::localfs" in parts
        assert "responses=inline::builtin" in parts
        assert len(parts) == 7  # 1 inference + 6 inline

    @patch("ogx.cli.stack.lets_go._probe_endpoint")
    def test_autodetect_uses_env_var_base_url(self, mock_probe: MagicMock, monkeypatch: pytest.MonkeyPatch):
        from ogx.cli.stack.lets_go import _autodetect_providers

        monkeypatch.setenv("OLLAMA_URL", "http://myhost:11434/v1")
        captured: list[str] = []

        def side_effect(
            base_url: str, probe_path: str, requires_key: bool, key_env: object, extra_headers: object = None
        ) -> _ProbeStatus:
            captured.append(base_url)
            return _ProbeStatus.UNREACHABLE

        mock_probe.side_effect = side_effect
        _autodetect_providers()
        assert captured[0] == "http://myhost:11434/v1"

    @patch("ogx.cli.stack.lets_go._probe_endpoint")
    def test_autodetect_result_order_matches_candidate_order(
        self, mock_probe: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        from ogx.cli.stack.lets_go import _autodetect_providers

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        def side_effect(
            base_url: str, probe_path: str, requires_key: bool, key_env: object, extra_headers: object = None
        ) -> _ProbeStatus:
            if "11434" in base_url or "openai.com" in base_url:
                return _ProbeStatus.OK
            return _ProbeStatus.UNREACHABLE

        mock_probe.side_effect = side_effect
        parts = _autodetect_providers().split(",")
        assert parts.index("inference=remote::ollama") < parts.index("inference=remote::openai")


class TestRunCommand:
    def test_no_inference_provider_exits(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args([])
        with (
            patch(
                "ogx.cli.stack.lets_go._autodetect_providers",
                return_value="files=inline::localfs,vector_io=inline::faiss,tool_runtime=inline::file-search,responses=inline::builtin",
            ),
            warnings.catch_warnings(),
            pytest.raises(SystemExit),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            lets_go._run_stack_lets_go_cmd(args)

    def test_empty_spec_exits(self, lets_go: StackLetsGo):
        args = lets_go.parser.parse_args([])
        with (
            patch("ogx.cli.stack.lets_go._autodetect_providers", return_value=""),
            warnings.catch_warnings(),
            pytest.raises(SystemExit),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            lets_go._run_stack_lets_go_cmd(args)

    @patch("ogx.cli.stack.lets_go._uvicorn_run")
    @patch("ogx.cli.stack.lets_go.get_provider_dependencies", return_value=([], [], []))
    @patch("ogx.cli.stack.lets_go.run_config_from_dynamic_config_spec")
    def test_providers_override_skips_autodetect(
        self,
        mock_build_config: MagicMock,
        mock_get_deps: MagicMock,
        mock_uvicorn_run: MagicMock,
        lets_go: StackLetsGo,
    ):
        args = lets_go.parser.parse_args(["--providers-override", "inference=remote::ollama"])
        mock_cfg = MagicMock()
        mock_cfg.model_dump.return_value = {}
        mock_build_config.return_value = mock_cfg

        with (
            patch("ogx.cli.stack.lets_go._autodetect_providers") as mock_detect,
            patch("builtins.open", MagicMock()),
            patch("ogx.cli.stack.lets_go.yaml.dump"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            lets_go._run_stack_lets_go_cmd(args)
        mock_detect.assert_not_called()

    @patch("ogx.cli.stack.lets_go._uvicorn_run")
    @patch("ogx.cli.stack.lets_go.get_provider_dependencies", return_value=([], [], []))
    @patch("ogx.cli.stack.lets_go.run_config_from_dynamic_config_spec")
    def test_run_command_uses_autodetected_providers(
        self,
        mock_build_config: MagicMock,
        mock_get_deps: MagicMock,
        mock_uvicorn_run: MagicMock,
        lets_go: StackLetsGo,
    ):
        args = lets_go.parser.parse_args([])
        mock_cfg = MagicMock()
        mock_cfg.model_dump.return_value = {}
        mock_build_config.return_value = mock_cfg

        with (
            patch("ogx.cli.stack.lets_go._autodetect_providers", return_value="inference=remote::ollama"),
            patch("builtins.open", MagicMock()),
            patch("ogx.cli.stack.lets_go.yaml.dump"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            lets_go._run_stack_lets_go_cmd(args)

        mock_build_config.assert_called_once()
        assert mock_build_config.call_args.kwargs["dynamic_config_spec"] == "inference=remote::ollama"

    @patch("ogx.cli.stack.lets_go._uvicorn_run")
    @patch("ogx.cli.stack.lets_go.subprocess.run")
    @patch("ogx.cli.stack.lets_go.get_provider_dependencies", return_value=(["httpx", "faiss-cpu"], [], []))
    @patch("ogx.cli.stack.lets_go.run_config_from_dynamic_config_spec")
    def test_install_deps_called_by_default(
        self,
        mock_build_config: MagicMock,
        mock_get_deps: MagicMock,
        mock_subprocess: MagicMock,
        mock_uvicorn_run: MagicMock,
        lets_go: StackLetsGo,
    ):
        args = lets_go.parser.parse_args([])
        mock_cfg = MagicMock()
        mock_cfg.model_dump.return_value = {}
        mock_build_config.return_value = mock_cfg
        mock_subprocess.return_value = MagicMock(returncode=0)

        with (
            patch("ogx.cli.stack.lets_go._autodetect_providers", return_value="inference=remote::ollama"),
            patch("builtins.open", MagicMock()),
            patch("ogx.cli.stack.lets_go.yaml.dump"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            lets_go._run_stack_lets_go_cmd(args)

        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "httpx" in call_args
        assert "faiss-cpu" in call_args

    @patch("ogx.cli.stack.lets_go._uvicorn_run")
    @patch("ogx.cli.stack.lets_go.subprocess.run")
    @patch("ogx.cli.stack.lets_go.get_provider_dependencies", return_value=(["httpx"], [], []))
    @patch("ogx.cli.stack.lets_go.run_config_from_dynamic_config_spec")
    def test_install_deps_skipped_with_flag(
        self,
        mock_build_config: MagicMock,
        mock_get_deps: MagicMock,
        mock_subprocess: MagicMock,
        mock_uvicorn_run: MagicMock,
        lets_go: StackLetsGo,
    ):
        args = lets_go.parser.parse_args(["--skip-install-deps"])
        mock_cfg = MagicMock()
        mock_cfg.model_dump.return_value = {}
        mock_build_config.return_value = mock_cfg

        with (
            patch("ogx.cli.stack.lets_go._autodetect_providers", return_value="inference=remote::ollama"),
            patch("builtins.open", MagicMock()),
            patch("ogx.cli.stack.lets_go.yaml.dump"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            lets_go._run_stack_lets_go_cmd(args)

        mock_subprocess.assert_not_called()


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
        assert "ogx letsgo" in str(future_warnings[0].message)

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

    def test_aliases_have_unprefixed_metadata(self):
        spec = "inference=remote::anthropic"
        aliases = _build_claude_code_aliases(spec)
        for alias in aliases:
            assert alias.metadata is not None
            assert alias.metadata.get("_unprefixed_alias") is True

    def test_priority_list_covers_expected_providers(self):
        assert "anthropic" in _CLAUDE_CODE_PROVIDER_PRIORITY
        assert "ollama" in _CLAUDE_CODE_PROVIDER_PRIORITY
        assert _CLAUDE_CODE_PROVIDER_PRIORITY.index("anthropic") < _CLAUDE_CODE_PROVIDER_PRIORITY.index("ollama")
