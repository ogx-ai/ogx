# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for `ogx connect claude` CLI command."""

import argparse
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from ogx.cli.connect.claude import ConnectClaude, _detect_tier_models, _strip_leading_separator


@pytest.fixture
def connect_claude() -> ConnectClaude:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return ConnectClaude(subparsers)


def _make_model(model_id: str, model_type: str = "llm") -> MagicMock:
    model = MagicMock()
    model.id = model_id
    model.model_extra = {"custom_metadata": {"model_type": model_type}}
    return model


def _make_mock_client(models: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.data = models
    client.models.list.return_value = response
    return client


class TestArguments:
    def test_defaults(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args([])
        assert args.url == "http://localhost:8321"
        assert args.model is None
        assert args.haiku_model is None
        assert args.sonnet_model is None
        assert args.opus_model is None
        assert args.print_env is False
        assert args.claude_args == []

    def test_url_override(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--url", "https://ogx.example.com"])
        assert args.url == "https://ogx.example.com"

    def test_model_override(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--model", "gpt-4o"])
        assert args.model == "gpt-4o"

    def test_haiku_model_override(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--haiku-model", "gpt-4o-mini"])
        assert args.haiku_model == "gpt-4o-mini"

    def test_sonnet_model_override(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--sonnet-model", "gpt-4o"])
        assert args.sonnet_model == "gpt-4o"

    def test_opus_model_override(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--opus-model", "o1"])
        assert args.opus_model == "o1"

    def test_print_env_flag(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--print-env"])
        assert args.print_env is True

    def test_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OGX_PORT", "9999")
        subparsers = argparse.ArgumentParser().add_subparsers()
        instance = ConnectClaude(subparsers)
        args = instance.parser.parse_args([])
        assert args.url == "http://localhost:9999"

    def test_claude_args_after_separator(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--", "-p", "hello world"])
        assert args.claude_args == ["--", "-p", "hello world"]


class TestClaudeDetection:
    def test_exits_when_claude_not_in_path(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args([])
        with patch("ogx.cli.connect.claude.shutil.which", return_value=None):
            with pytest.raises(SystemExit):
                connect_claude._run_connect_claude_cmd(args)

    def test_continues_when_claude_found(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.connect.claude.shutil.which", return_value="/usr/bin/claude"),
            patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.claude.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit) as exc_info:
                connect_claude._run_connect_claude_cmd(args)
            assert exc_info.value.code == 0


class TestServerProbe:
    def test_exits_when_server_unreachable(self, connect_claude: ConnectClaude) -> None:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = APIConnectionError(request=httpx.Request("GET", "http://localhost"))

        with patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                connect_claude._fetch_models("http://localhost:8321/v1")

    def test_exits_on_server_error(self, connect_claude: ConnectClaude) -> None:
        mock_client = MagicMock()
        mock_response = httpx.Response(500, request=httpx.Request("GET", "http://localhost"))
        mock_client.models.list.side_effect = APIStatusError("server error", response=mock_response, body=None)

        with patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                connect_claude._fetch_models("http://localhost:8321/v1")

    def test_returns_models_on_success(self, connect_claude: ConnectClaude) -> None:
        mock_client = _make_mock_client([_make_model("gpt-4o"), _make_model("llama-3.1-8b")])

        with patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client):
            models = connect_claude._fetch_models("http://localhost:8321/v1")
        assert models == ["gpt-4o", "llama-3.1-8b"]

    def test_filters_out_embedding_models(self, connect_claude: ConnectClaude) -> None:
        mock_client = _make_mock_client(
            [
                _make_model("gpt-4o"),
                _make_model("text-embedding-3-small", model_type="embedding"),
            ]
        )

        with patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client):
            models = connect_claude._fetch_models("http://localhost:8321/v1")
        assert "text-embedding-3-small" not in models
        assert "gpt-4o" in models


class TestModelMapping:
    def test_all_tiers_default_to_first_model_when_no_keywords(self, connect_claude: ConnectClaude) -> None:
        mapping = connect_claude._resolve_model_mapping(
            model=None,
            haiku_model=None,
            sonnet_model=None,
            opus_model=None,
            available_models=["gpt-4o", "llama-3.1-8b"],
        )
        assert mapping["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "gpt-4o"
        assert mapping["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "gpt-4o"
        assert mapping["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "gpt-4o"

    def test_auto_detects_tiers_from_model_names(self, connect_claude: ConnectClaude) -> None:
        mapping = connect_claude._resolve_model_mapping(
            model=None,
            haiku_model=None,
            sonnet_model=None,
            opus_model=None,
            available_models=["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"],
        )
        assert mapping["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-haiku-4-5"
        assert mapping["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-sonnet-4-6"
        assert mapping["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "claude-opus-4-7"

    def test_auto_detects_partial_tiers(self, connect_claude: ConnectClaude) -> None:
        mapping = connect_claude._resolve_model_mapping(
            model=None,
            haiku_model=None,
            sonnet_model=None,
            opus_model=None,
            available_models=["claude-sonnet-4-6", "llama-3.3-70b"],
        )
        assert mapping["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-sonnet-4-6"
        assert mapping["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-sonnet-4-6"
        assert mapping["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "claude-sonnet-4-6"

    def test_model_flag_sets_all_tiers(self, connect_claude: ConnectClaude) -> None:
        mapping = connect_claude._resolve_model_mapping(
            model="llama-3.1-8b",
            haiku_model=None,
            sonnet_model=None,
            opus_model=None,
            available_models=["gpt-4o", "llama-3.1-8b"],
        )
        assert mapping["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "llama-3.1-8b"
        assert mapping["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "llama-3.1-8b"
        assert mapping["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "llama-3.1-8b"

    def test_per_tier_overrides_model_flag(self, connect_claude: ConnectClaude) -> None:
        mapping = connect_claude._resolve_model_mapping(
            model="gpt-4o",
            haiku_model="llama-3.1-8b",
            sonnet_model=None,
            opus_model=None,
            available_models=["gpt-4o", "llama-3.1-8b"],
        )
        assert mapping["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "llama-3.1-8b"
        assert mapping["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "gpt-4o"
        assert mapping["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "gpt-4o"

    def test_all_tiers_individually_set(self, connect_claude: ConnectClaude) -> None:
        mapping = connect_claude._resolve_model_mapping(
            model=None,
            haiku_model="gpt-4o-mini",
            sonnet_model="gpt-4o",
            opus_model="o1",
            available_models=["gpt-4o-mini", "gpt-4o", "o1"],
        )
        assert mapping["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "gpt-4o-mini"
        assert mapping["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "gpt-4o"
        assert mapping["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "o1"

    def test_exits_when_model_not_available(self, connect_claude: ConnectClaude) -> None:
        with pytest.raises(SystemExit):
            connect_claude._resolve_model_mapping(
                model="nonexistent", haiku_model=None, sonnet_model=None, opus_model=None, available_models=["gpt-4o"]
            )

    def test_exits_when_tier_model_not_available(self, connect_claude: ConnectClaude) -> None:
        with pytest.raises(SystemExit):
            connect_claude._resolve_model_mapping(
                model=None, haiku_model="nonexistent", sonnet_model=None, opus_model=None, available_models=["gpt-4o"]
            )

    def test_model_ids_with_slashes(self, connect_claude: ConnectClaude) -> None:
        mapping = connect_claude._resolve_model_mapping(
            model="ollama/llama3.3:70b",
            haiku_model=None,
            sonnet_model=None,
            opus_model=None,
            available_models=["ollama/llama3.3:70b"],
        )
        assert mapping["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "ollama/llama3.3:70b"


class TestEnvironment:
    def test_sets_anthropic_base_url(self, connect_claude: ConnectClaude) -> None:
        env = connect_claude._build_env("http://localhost:8321", {})
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:8321"

    def test_base_url_has_no_v1_suffix(self, connect_claude: ConnectClaude) -> None:
        env = connect_claude._build_env("http://localhost:8321", {})
        assert not env["ANTHROPIC_BASE_URL"].endswith("/v1")

    def test_sets_anthropic_auth_token(self, connect_claude: ConnectClaude) -> None:
        env = connect_claude._build_env("http://localhost:8321", {})
        assert env["ANTHROPIC_AUTH_TOKEN"] == "ogx"

    def test_sets_model_tier_env_vars(self, connect_claude: ConnectClaude) -> None:
        mapping = {
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-4o-mini",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-4o",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "o1",
        }
        env = connect_claude._build_env("http://localhost:8321", mapping)
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "gpt-4o-mini"
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "gpt-4o"
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "o1"

    def test_unsets_vertex_vars(self, connect_claude: ConnectClaude, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
        monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "my-project")
        env = connect_claude._build_env("http://localhost:8321", {})
        assert "CLAUDE_CODE_USE_VERTEX" not in env
        assert "ANTHROPIC_VERTEX_PROJECT_ID" not in env

    def test_unsets_bedrock_vars(self, connect_claude: ConnectClaude, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
        monkeypatch.setenv("ANTHROPIC_BEDROCK_SESSION_TOKEN", "tok")
        env = connect_claude._build_env("http://localhost:8321", {})
        assert "CLAUDE_CODE_USE_BEDROCK" not in env
        assert "ANTHROPIC_BEDROCK_SESSION_TOKEN" not in env

    def test_preserves_existing_env_vars(self, connect_claude: ConnectClaude, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_CUSTOM_VAR", "hello")
        env = connect_claude._build_env("http://localhost:8321", {})
        assert env["MY_CUSTOM_VAR"] == "hello"


class TestPrintEnv:
    def test_prints_export_statements(self, connect_claude: ConnectClaude, capsys: pytest.CaptureFixture[str]) -> None:
        args = connect_claude.parser.parse_args(["--print-env", "--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client):
            connect_claude._run_connect_claude_cmd(args)

        output = capsys.readouterr().out
        assert "export ANTHROPIC_BASE_URL=http://localhost:8321" in output
        assert "export ANTHROPIC_AUTH_TOKEN=ogx" in output
        assert "export ANTHROPIC_DEFAULT_HAIKU_MODEL=gpt-4o" in output
        assert "export ANTHROPIC_DEFAULT_SONNET_MODEL=gpt-4o" in output
        assert "export ANTHROPIC_DEFAULT_OPUS_MODEL=gpt-4o" in output
        assert "unset CLAUDE_CODE_USE_VERTEX" in output
        assert "unset CLAUDE_CODE_USE_BEDROCK" in output

    def test_prints_https_url(self, connect_claude: ConnectClaude, capsys: pytest.CaptureFixture[str]) -> None:
        args = connect_claude.parser.parse_args(
            ["--print-env", "--url", "https://ogx.example.com", "--model", "gpt-4o"]
        )
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client):
            connect_claude._run_connect_claude_cmd(args)

        output = capsys.readouterr().out
        assert "export ANTHROPIC_BASE_URL=https://ogx.example.com" in output

    def test_auto_detects_models_in_print_env(
        self, connect_claude: ConnectClaude, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = connect_claude.parser.parse_args(["--print-env"])
        mock_client = _make_mock_client(
            [_make_model("claude-haiku-4-5"), _make_model("claude-sonnet-4-6"), _make_model("claude-opus-4-7")]
        )

        with patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client):
            connect_claude._run_connect_claude_cmd(args)

        output = capsys.readouterr().out
        assert "export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-4-5" in output
        assert "export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-4-6" in output
        assert "export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-4-7" in output

    def test_does_not_launch_subprocess(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--print-env", "--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.claude.subprocess.run") as mock_run,
        ):
            connect_claude._run_connect_claude_cmd(args)
            mock_run.assert_not_called()

    def test_skips_claude_binary_check(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--print-env", "--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.connect.claude.shutil.which", return_value=None) as mock_which,
            patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client),
        ):
            connect_claude._run_connect_claude_cmd(args)
            mock_which.assert_not_called()


class TestConnect:
    def test_launches_claude_with_env(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.connect.claude.shutil.which", return_value="/usr/bin/claude"),
            patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.claude.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit):
                connect_claude._run_connect_claude_cmd(args)

            mock_run.assert_called_once()
            call_env = mock_run.call_args.kwargs["env"]
            assert call_env["ANTHROPIC_BASE_URL"] == "http://localhost:8321"
            assert call_env["ANTHROPIC_AUTH_TOKEN"] == "ogx"
            assert call_env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "gpt-4o"

    def test_forwards_extra_args(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--model", "gpt-4o", "--", "-p", "hello"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.connect.claude.shutil.which", return_value="/usr/bin/claude"),
            patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.claude.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit):
                connect_claude._run_connect_claude_cmd(args)

            cmd = mock_run.call_args.args[0]
            assert cmd == ["claude", "-p", "hello"]

    def test_propagates_exit_code(self, connect_claude: ConnectClaude) -> None:
        args = connect_claude.parser.parse_args(["--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.connect.claude.shutil.which", return_value="/usr/bin/claude"),
            patch("ogx.cli.connect.claude.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.claude.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=42)
            with pytest.raises(SystemExit) as exc_info:
                connect_claude._run_connect_claude_cmd(args)
            assert exc_info.value.code == 42


class TestDetectTierModels:
    def test_detects_all_tiers(self) -> None:
        result = _detect_tier_models(["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"])
        assert result == {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-7"}

    def test_detects_partial_tiers(self) -> None:
        result = _detect_tier_models(["claude-sonnet-4-6", "llama-3.3-70b"])
        assert result == {"sonnet": "claude-sonnet-4-6"}

    def test_no_matches(self) -> None:
        result = _detect_tier_models(["gpt-4o", "llama-3.3-70b"])
        assert result == {}

    def test_picks_first_match_per_tier(self) -> None:
        result = _detect_tier_models(["claude-haiku-4-5", "claude-haiku-4-5-20251001"])
        assert result == {"haiku": "claude-haiku-4-5"}

    def test_case_insensitive(self) -> None:
        result = _detect_tier_models(["Claude-Haiku-4-5"])
        assert result == {"haiku": "Claude-Haiku-4-5"}


class TestStripLeadingSeparator:
    def test_strips_leading_double_dash(self) -> None:
        assert _strip_leading_separator(["--", "-p", "hello"]) == ["-p", "hello"]

    def test_preserves_args_without_separator(self) -> None:
        assert _strip_leading_separator(["-p", "hello"]) == ["-p", "hello"]

    def test_empty_list(self) -> None:
        assert _strip_leading_separator([]) == []

    def test_only_separator(self) -> None:
        assert _strip_leading_separator(["--"]) == []
