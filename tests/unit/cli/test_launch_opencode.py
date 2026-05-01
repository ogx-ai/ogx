# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for `ogx launch opencode` CLI command."""

import argparse
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from ogx.cli.launch.opencode import LaunchOpenCode


@pytest.fixture
def launch_opencode() -> LaunchOpenCode:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return LaunchOpenCode(subparsers)


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
    def test_defaults(self, launch_opencode: LaunchOpenCode) -> None:
        args = launch_opencode.parser.parse_args([])
        assert args.port == 8321
        assert args.host == "localhost"
        assert args.model is None

    def test_port_override(self, launch_opencode: LaunchOpenCode) -> None:
        args = launch_opencode.parser.parse_args(["--port", "9000"])
        assert args.port == 9000

    def test_host_override(self, launch_opencode: LaunchOpenCode) -> None:
        args = launch_opencode.parser.parse_args(["--host", "0.0.0.0"])
        assert args.host == "0.0.0.0"

    def test_model_override(self, launch_opencode: LaunchOpenCode) -> None:
        args = launch_opencode.parser.parse_args(["--model", "gpt-4o"])
        assert args.model == "gpt-4o"

    def test_port_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OGX_PORT", "9999")
        subparsers = argparse.ArgumentParser().add_subparsers()
        instance = LaunchOpenCode(subparsers)
        args = instance.parser.parse_args([])
        assert args.port == 9999


class TestOpenCodeDetection:
    def test_exits_when_opencode_not_in_path(self, launch_opencode: LaunchOpenCode) -> None:
        args = launch_opencode.parser.parse_args([])
        with patch("ogx.cli.launch.opencode.shutil.which", return_value=None):
            with pytest.raises(SystemExit):
                launch_opencode._run_launch_opencode_cmd(args)

    def test_continues_when_opencode_found(self, launch_opencode: LaunchOpenCode) -> None:
        args = launch_opencode.parser.parse_args(["--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.launch.opencode.shutil.which", return_value="/usr/bin/opencode"),
            patch("ogx.cli.launch.opencode.OpenAI", return_value=mock_client),
            patch("ogx.cli.launch.opencode.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit) as exc_info:
                launch_opencode._run_launch_opencode_cmd(args)
            assert exc_info.value.code == 0


class TestServerProbe:
    def test_exits_when_server_unreachable(self, launch_opencode: LaunchOpenCode) -> None:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = APIConnectionError(request=httpx.Request("GET", "http://localhost"))

        with patch("ogx.cli.launch.opencode.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                launch_opencode._fetch_models("http://localhost:8321/v1")

    def test_exits_on_timeout(self, launch_opencode: LaunchOpenCode) -> None:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = APIConnectionError(request=httpx.Request("GET", "http://localhost"))

        with patch("ogx.cli.launch.opencode.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                launch_opencode._fetch_models("http://localhost:8321/v1")

    def test_exits_on_server_error(self, launch_opencode: LaunchOpenCode) -> None:
        mock_client = MagicMock()
        mock_response = httpx.Response(500, request=httpx.Request("GET", "http://localhost"))
        mock_client.models.list.side_effect = APIStatusError("server error", response=mock_response, body=None)

        with patch("ogx.cli.launch.opencode.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                launch_opencode._fetch_models("http://localhost:8321/v1")

    def test_returns_models_on_success(self, launch_opencode: LaunchOpenCode) -> None:
        mock_client = _make_mock_client([_make_model("gpt-4o"), _make_model("llama-3.1-8b")])

        with patch("ogx.cli.launch.opencode.OpenAI", return_value=mock_client):
            models = launch_opencode._fetch_models("http://localhost:8321/v1")
        assert models == ["gpt-4o", "llama-3.1-8b"]


class TestModelSelection:
    def test_uses_specified_model(self, launch_opencode: LaunchOpenCode) -> None:
        result = launch_opencode._select_default_model("gpt-4o", ["gpt-4o", "llama-3.1-8b"])
        assert result == "gpt-4o"

    def test_exits_when_specified_model_not_found(self, launch_opencode: LaunchOpenCode) -> None:
        with pytest.raises(SystemExit):
            launch_opencode._select_default_model("nonexistent", ["gpt-4o", "llama-3.1-8b"])

    def test_defaults_to_first_model(self, launch_opencode: LaunchOpenCode) -> None:
        result = launch_opencode._select_default_model(None, ["gpt-4o", "llama-3.1-8b"])
        assert result == "gpt-4o"

    def test_filters_out_embedding_models(self, launch_opencode: LaunchOpenCode) -> None:
        mock_client = _make_mock_client(
            [
                _make_model("gpt-4o"),
                _make_model("text-embedding-3-small", model_type="embedding"),
            ]
        )

        with patch("ogx.cli.launch.opencode.OpenAI", return_value=mock_client):
            models = launch_opencode._fetch_models("http://localhost:8321/v1")
        assert "text-embedding-3-small" not in models
        assert "gpt-4o" in models

    def test_exits_when_no_llm_models(self, launch_opencode: LaunchOpenCode) -> None:
        mock_client = _make_mock_client(
            [
                _make_model("text-embedding-3-small", model_type="embedding"),
            ]
        )

        with patch("ogx.cli.launch.opencode.OpenAI", return_value=mock_client):
            models = launch_opencode._fetch_models("http://localhost:8321/v1")
        assert models == []


class TestConfigGeneration:
    def test_config_structure(self, launch_opencode: LaunchOpenCode) -> None:
        config = launch_opencode._build_opencode_config(
            "http://localhost:8321/v1", ["gpt-4o", "llama-3.1-8b"], "gpt-4o"
        )
        assert config["$schema"] == "https://opencode.ai/config.json"
        assert config["model"] == "ogx/gpt-4o"
        assert "ogx" in config["provider"]
        provider = config["provider"]["ogx"]
        assert provider["npm"] == "@ai-sdk/openai-compatible"
        assert provider["name"] == "OGX"
        assert provider["options"]["baseURL"] == "http://localhost:8321/v1"
        assert "gpt-4o" in provider["models"]
        assert "llama-3.1-8b" in provider["models"]
        model = provider["models"]["gpt-4o"]
        assert model["tools"] is True
        assert model["limit"]["context"] == 128000
        assert model["limit"]["output"] == 4096

    def test_config_includes_all_models(self, launch_opencode: LaunchOpenCode) -> None:
        all_models = ["gpt-4o", "llama-3.1-8b", "claude-3.5-sonnet"]
        config = launch_opencode._build_opencode_config("http://localhost:8321/v1", all_models, "gpt-4o")
        provider_models = config["provider"]["ogx"]["models"]
        assert set(provider_models.keys()) == set(all_models)

    def test_config_uses_correct_base_url(self, launch_opencode: LaunchOpenCode) -> None:
        config = launch_opencode._build_opencode_config("http://myhost:9000/v1", ["llama-3.1-8b"], "llama-3.1-8b")
        assert config["provider"]["ogx"]["options"]["baseURL"] == "http://myhost:9000/v1"
        assert "llama-3.1-8b" in config["provider"]["ogx"]["models"]

    def test_config_is_valid_json(self, launch_opencode: LaunchOpenCode) -> None:
        config = launch_opencode._build_opencode_config("http://localhost:8321/v1", ["gpt-4o"], "gpt-4o")
        serialized = json.dumps(config)
        parsed = json.loads(serialized)
        assert parsed == config


class TestLaunch:
    def test_launches_opencode_with_config_env(self, launch_opencode: LaunchOpenCode) -> None:
        args = launch_opencode.parser.parse_args(["--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.launch.opencode.shutil.which", return_value="/usr/bin/opencode"),
            patch("ogx.cli.launch.opencode.OpenAI", return_value=mock_client),
            patch("ogx.cli.launch.opencode.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit):
                launch_opencode._run_launch_opencode_cmd(args)

            mock_run.assert_called_once()
            call_env = mock_run.call_args.kwargs["env"]
            assert "OPENCODE_CONFIG_CONTENT" in call_env
            config = json.loads(call_env["OPENCODE_CONFIG_CONTENT"])
            assert config["provider"]["ogx"]["options"]["baseURL"] == "http://localhost:8321/v1"

    def test_propagates_exit_code(self, launch_opencode: LaunchOpenCode) -> None:
        args = launch_opencode.parser.parse_args(["--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.launch.opencode.shutil.which", return_value="/usr/bin/opencode"),
            patch("ogx.cli.launch.opencode.OpenAI", return_value=mock_client),
            patch("ogx.cli.launch.opencode.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=42)
            with pytest.raises(SystemExit) as exc_info:
                launch_opencode._run_launch_opencode_cmd(args)
            assert exc_info.value.code == 42
