# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for `ogx connect opencode` CLI command."""

import argparse
import json
from unittest.mock import MagicMock, patch

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError

from ogx.cli.connect.opencode import ConnectOpenCode

from .connect_helpers import make_untrusted_certificate_error, parse_group_args


@pytest.fixture
def connect_opencode() -> ConnectOpenCode:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return ConnectOpenCode(subparsers)


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
    def test_defaults(self, connect_opencode: ConnectOpenCode, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OGX_BASE_URL", raising=False)
        args = parse_group_args(["opencode"])
        assert args.model is None
        assert args.connect_url == "http://localhost:8321/v1"

    def test_group_url_is_used(self) -> None:
        args = parse_group_args(["--url", "https://group.example.com/v1", "opencode"])
        assert args.connect_url == "https://group.example.com/v1"

    def test_agent_level_host_port_flags_removed(self, connect_opencode: ConnectOpenCode) -> None:
        with pytest.raises(SystemExit):
            parse_group_args(["opencode", "--host", "0.0.0.0"])
        with pytest.raises(SystemExit):
            parse_group_args(["opencode", "--port", "9000"])

    def test_model_override(self, connect_opencode: ConnectOpenCode) -> None:
        args = parse_group_args(["opencode", "--model", "gpt-4o"])
        assert args.model == "gpt-4o"

    def test_port_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OGX_BASE_URL", raising=False)
        monkeypatch.setenv("OGX_PORT", "9999")
        args = parse_group_args(["opencode"])
        assert args.connect_url == "http://localhost:9999/v1"


class TestOpenCodeDetection:
    def test_exits_when_opencode_not_in_path(self, connect_opencode: ConnectOpenCode) -> None:
        args = parse_group_args(
            [
                "opencode",
            ]
        )
        with patch("ogx.cli.connect.opencode.shutil.which", return_value=None):
            with pytest.raises(SystemExit):
                connect_opencode._run_connect_opencode_cmd(args)

    def test_continues_when_opencode_found(self, connect_opencode: ConnectOpenCode) -> None:
        args = parse_group_args(["opencode", "--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.connect.opencode.shutil.which", return_value="/usr/bin/opencode"),
            patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.opencode.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit) as exc_info:
                connect_opencode._run_connect_opencode_cmd(args)
            assert exc_info.value.code == 0


class TestServerProbe:
    def test_exits_when_server_unreachable(self, connect_opencode: ConnectOpenCode) -> None:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = APIConnectionError(request=httpx2.Request("GET", "http://localhost"))

        with patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                connect_opencode._fetch_models("http://localhost:8321/v1", False)

    def test_exits_on_timeout(self, connect_opencode: ConnectOpenCode) -> None:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = APIConnectionError(request=httpx2.Request("GET", "http://localhost"))

        with patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                connect_opencode._fetch_models("http://localhost:8321/v1", False)

    def test_exits_on_server_error(self, connect_opencode: ConnectOpenCode) -> None:
        mock_client = MagicMock()
        mock_response = httpx2.Response(500, request=httpx2.Request("GET", "http://localhost"))
        mock_client.models.list.side_effect = APIStatusError("server error", response=mock_response, body=None)

        with patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                connect_opencode._fetch_models("http://localhost:8321/v1", False)

    def test_returns_models_on_success(self, connect_opencode: ConnectOpenCode) -> None:
        mock_client = _make_mock_client([_make_model("gpt-4o"), _make_model("llama-3.1-8b")])

        with patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client):
            models, _ = connect_opencode._fetch_models("http://localhost:8321/v1", False)
        assert models == ["gpt-4o", "llama-3.1-8b"]


class TestModelSelection:
    def test_uses_specified_model(self, connect_opencode: ConnectOpenCode) -> None:
        result = connect_opencode._select_default_model("gpt-4o", ["gpt-4o", "llama-3.1-8b"])
        assert result == "gpt-4o"

    def test_exits_when_specified_model_not_found(self, connect_opencode: ConnectOpenCode) -> None:
        with pytest.raises(SystemExit):
            connect_opencode._select_default_model("nonexistent", ["gpt-4o", "llama-3.1-8b"])

    def test_defaults_to_first_model(self, connect_opencode: ConnectOpenCode) -> None:
        result = connect_opencode._select_default_model(None, ["gpt-4o", "llama-3.1-8b"])
        assert result == "gpt-4o"

    def test_filters_out_embedding_models(self, connect_opencode: ConnectOpenCode) -> None:
        mock_client = _make_mock_client(
            [
                _make_model("gpt-4o"),
                _make_model("text-embedding-3-small", model_type="embedding"),
            ]
        )

        with patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client):
            models, _ = connect_opencode._fetch_models("http://localhost:8321/v1", False)
        assert "text-embedding-3-small" not in models
        assert "gpt-4o" in models

    def test_exits_when_no_llm_models(self, connect_opencode: ConnectOpenCode) -> None:
        mock_client = _make_mock_client(
            [
                _make_model("text-embedding-3-small", model_type="embedding"),
            ]
        )

        with patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client):
            models, _ = connect_opencode._fetch_models("http://localhost:8321/v1", False)
        assert models == []


class TestConfigGeneration:
    def test_config_structure(self, connect_opencode: ConnectOpenCode) -> None:
        config = connect_opencode._build_opencode_config(
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

    def test_config_includes_all_models(self, connect_opencode: ConnectOpenCode) -> None:
        all_models = ["gpt-4o", "llama-3.1-8b", "claude-3.5-sonnet"]
        config = connect_opencode._build_opencode_config("http://localhost:8321/v1", all_models, "gpt-4o")
        provider_models = config["provider"]["ogx"]["models"]
        assert set(provider_models.keys()) == set(all_models)

    def test_config_uses_correct_base_url(self, connect_opencode: ConnectOpenCode) -> None:
        config = connect_opencode._build_opencode_config("http://myhost:9000/v1", ["llama-3.1-8b"], "llama-3.1-8b")
        assert config["provider"]["ogx"]["options"]["baseURL"] == "http://myhost:9000/v1"
        assert "llama-3.1-8b" in config["provider"]["ogx"]["models"]

    def test_config_is_valid_json(self, connect_opencode: ConnectOpenCode) -> None:
        config = connect_opencode._build_opencode_config("http://localhost:8321/v1", ["gpt-4o"], "gpt-4o")
        serialized = json.dumps(config)
        parsed = json.loads(serialized)
        assert parsed == config


class TestSelfSignedTLS:
    def test_untrusted_cert_without_insecure_exits_with_hint(
        self, connect_opencode: ConnectOpenCode, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = make_untrusted_certificate_error()

        with patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                connect_opencode._fetch_models("https://ogx.example.com/v1", False)
        output = capsys.readouterr().err
        assert "TLS certificate is not trusted" in output
        assert "--insecure" in output

    def test_self_signed_exports_tls_env_for_launched_process(self, connect_opencode: ConnectOpenCode) -> None:
        mock_client = _make_mock_client([_make_model("gpt-4o")])
        mock_client.models.list.side_effect = [make_untrusted_certificate_error(), mock_client.models.list.return_value]
        args = parse_group_args(["-k", "opencode"])

        with (
            patch("ogx.cli.connect.opencode.shutil.which", return_value="/usr/bin/opencode"),
            patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.opencode.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit):
                connect_opencode._run_connect_opencode_cmd(args)

            call_env = mock_run.call_args.kwargs["env"]
            assert call_env["NODE_TLS_REJECT_UNAUTHORIZED"] == "0"

    def test_trusted_server_does_not_export_tls_env(self, connect_opencode: ConnectOpenCode) -> None:
        mock_client = _make_mock_client([_make_model("gpt-4o")])
        args = parse_group_args(["opencode"])

        with (
            patch("ogx.cli.connect.opencode.shutil.which", return_value="/usr/bin/opencode"),
            patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.opencode.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit):
                connect_opencode._run_connect_opencode_cmd(args)

            assert "NODE_TLS_REJECT_UNAUTHORIZED" not in mock_run.call_args.kwargs["env"]


class TestConnect:
    def test_connects_opencode_with_config_env(self, connect_opencode: ConnectOpenCode) -> None:
        args = parse_group_args(["opencode", "--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.connect.opencode.shutil.which", return_value="/usr/bin/opencode"),
            patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.opencode.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit):
                connect_opencode._run_connect_opencode_cmd(args)

            mock_run.assert_called_once()
            call_env = mock_run.call_args.kwargs["env"]
            assert "OPENCODE_CONFIG_CONTENT" in call_env
            config = json.loads(call_env["OPENCODE_CONFIG_CONTENT"])
            assert config["provider"]["ogx"]["options"]["baseURL"] == "http://localhost:8321/v1"

    def test_propagates_exit_code(self, connect_opencode: ConnectOpenCode) -> None:
        args = parse_group_args(["opencode", "--model", "gpt-4o"])
        mock_client = _make_mock_client([_make_model("gpt-4o")])

        with (
            patch("ogx.cli.connect.opencode.shutil.which", return_value="/usr/bin/opencode"),
            patch("ogx.cli.connect.opencode.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.opencode.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=42)
            with pytest.raises(SystemExit) as exc_info:
                connect_opencode._run_connect_opencode_cmd(args)
            assert exc_info.value.code == 42
