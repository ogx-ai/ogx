# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for `ogx letsgo` and `ogx stack letsgo` CLI commands."""

import argparse
import warnings
from unittest.mock import MagicMock, patch

import pytest

from ogx.cli.letsgo import LetsGo
from ogx.cli.stack.lets_go import (
    _CLAUDE_CODE_ALIASES,
    _CLAUDE_CODE_PROVIDER_PRIORITY,
    StackLetsGo,
    _build_claude_code_aliases,
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


class TestAutodetect:
    @patch(
        "ogx.cli.stack.lets_go._probe_provider_availability",
        return_value=(_ProbeStatus.UNREACHABLE, 0, "", "default", None),
    )
    def test_autodetect_no_providers(self, mock_probe: MagicMock):
        from ogx.cli.stack.lets_go import _autodetect_providers

        parts = _autodetect_providers().split(",")
        assert "files=inline::localfs" in parts
        assert "vector_io=inline::faiss" in parts
        assert "tool_runtime=inline::file-search" in parts
        assert "responses=inline::builtin" in parts

    @patch(
        "ogx.cli.stack.lets_go._probe_provider_availability", return_value=(_ProbeStatus.NO_KEY, 0, "", "default", None)
    )
    def test_no_key_providers_excluded(self, mock_probe: MagicMock):
        from ogx.cli.stack.lets_go import _autodetect_providers

        parts = _autodetect_providers().split(",")
        assert "files=inline::localfs" in parts
        assert "vector_io=inline::faiss" in parts
        assert "tool_runtime=inline::file-search" in parts
        assert "responses=inline::builtin" in parts

    @patch(
        "ogx.cli.stack.lets_go._probe_provider_availability",
        return_value=(_ProbeStatus.OK, 3, "http://test", "default", None),
    )
    def test_autodetect_all_ok(self, mock_probe: MagicMock):
        from ogx.cli.stack.lets_go import _autodetect_providers

        result = _autodetect_providers()
        parts = result.split(",")
        assert "inference=remote::ollama" in parts
        assert "inference=remote::anthropic" in parts
        assert "files=inline::localfs" in parts
        assert "responses=inline::builtin" in parts
        assert len(parts) == 14  # 8 probed + 6 inline

    @patch("ogx.cli.stack.lets_go._probe_provider_availability")
    def test_autodetect_only_ollama(self, mock_probe: MagicMock):
        from ogx.cli.stack.lets_go import _autodetect_providers

        def side_effect(
            provider_type: str,
            base_url_env: object,
            default_base_url: str,
            required_api_key_env: object,
            optional_api_key_env: object = None,
        ) -> tuple:
            if provider_type == "remote::ollama":
                return (_ProbeStatus.OK, 3, "http://localhost:11434/v1", "default", None)
            return (_ProbeStatus.UNREACHABLE, 0, "", "default", None)

        mock_probe.side_effect = side_effect
        parts = _autodetect_providers().split(",")
        assert "inference=remote::ollama" in parts
        assert "files=inline::localfs" in parts
        assert "responses=inline::builtin" in parts
        assert len(parts) == 7  # 1 inference + 6 inline

    @patch("ogx.cli.stack.lets_go._probe_provider_availability")
    def test_autodetect_uses_env_var_name(self, mock_probe: MagicMock, monkeypatch: pytest.MonkeyPatch):
        from ogx.cli.stack.lets_go import _autodetect_providers

        monkeypatch.setenv("OLLAMA_URL", "http://myhost:11434/v1")
        captured: list[str] = []

        def side_effect(
            provider_type: str,
            base_url_env: object,
            default_base_url: str,
            required_api_key_env: object,
            optional_api_key_env: object = None,
        ) -> tuple:
            if provider_type == "remote::ollama":
                captured.append(base_url_env)
            return (_ProbeStatus.UNREACHABLE, 0, "", "default", None)

        mock_probe.side_effect = side_effect
        _autodetect_providers()
        assert captured[0] == "OLLAMA_URL"

    @patch("ogx.cli.stack.lets_go._probe_provider_availability")
    def test_autodetect_result_order_matches_candidate_order(
        self, mock_probe: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        from ogx.cli.stack.lets_go import _autodetect_providers

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        def side_effect(
            provider_type: str,
            base_url_env: object,
            default_base_url: str,
            required_api_key_env: object,
            optional_api_key_env: object = None,
        ) -> tuple:
            if provider_type in ("remote::ollama", "remote::openai"):
                return (_ProbeStatus.OK, 3, "http://test", "default", None)
            return (_ProbeStatus.UNREACHABLE, 0, "", "default", None)

        mock_probe.side_effect = side_effect
        parts = _autodetect_providers().split(",")
        assert parts.index("inference=remote::ollama") < parts.index("inference=remote::openai")

    @patch("ogx.cli.stack.lets_go._probe_provider_availability")
    def test_autodetect_includes_vllm_on_needs_key(self, mock_probe: MagicMock, monkeypatch: pytest.MonkeyPatch):
        from ogx.cli.stack.lets_go import _autodetect_providers

        monkeypatch.delenv("VLLM_API_TOKEN", raising=False)

        def side_effect(
            provider_type: str,
            base_url_env: object,
            default_base_url: str,
            required_api_key_env: object,
            optional_api_key_env: object = None,
        ) -> tuple:
            if provider_type == "remote::vllm":
                return (_ProbeStatus.NEEDS_KEY, 3, "http://localhost:8000/v1", "default", None)
            return (_ProbeStatus.UNREACHABLE, 0, "", "default", None)

        mock_probe.side_effect = side_effect
        parts = _autodetect_providers().split(",")
        assert "inference=remote::vllm" in parts


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
