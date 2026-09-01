# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for `ogx connect codex` CLI command."""

import argparse
import contextlib
import json
import tomllib
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from ogx.cli.connect.codex import (
    CodexCatalogBuilder,
    CodexServerDiscovery,
    CodexSessionBuilder,
    ConnectCodex,
    DiscoveredCodexModel,
)


@pytest.fixture
def connect_codex() -> ConnectCodex:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return ConnectCodex(subparsers)


@pytest.fixture
def discovery(connect_codex: ConnectCodex) -> CodexServerDiscovery:
    return connect_codex.discovery


@pytest.fixture
def session_builder(connect_codex: ConnectCodex) -> CodexSessionBuilder:
    return connect_codex.session_builder


@pytest.fixture
def catalog_builder(session_builder: CodexSessionBuilder) -> CodexCatalogBuilder:
    return session_builder.catalog_builder


def _make_model(model_id: str, model_type: str = "llm", metadata: dict | None = None) -> MagicMock:
    custom_metadata = {"model_type": model_type}
    if metadata:
        custom_metadata.update(metadata)
    model = MagicMock()
    model.id = model_id
    model.model_extra = {"custom_metadata": custom_metadata}
    model.custom_metadata = custom_metadata
    return model


def _make_mock_client(models: list[MagicMock]) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.data = models
    client.models.list.return_value = response
    return client


class TestArguments:
    def test_defaults(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args([])
        assert args.url == connect_codex.DEFAULT_BASE_URL
        assert args.model is None

    def test_url_override(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args(["--url", "https://ogx.example.com/v1"])
        assert args.url == "https://ogx.example.com/v1"

    def test_model_override(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args(["--model", "openai/gpt-4o"])
        assert args.model == "openai/gpt-4o"

    def test_exec_prompt(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args(["--exec", "Reply with only Paris"])
        assert args.exec_prompt == "Reply with only Paris"

    def test_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OGX_BASE_URL", "https://ogx.example.com/custom/v1")
        subparsers = argparse.ArgumentParser().add_subparsers()
        instance = ConnectCodex(subparsers)
        args = instance.parser.parse_args([])
        assert args.url == "https://ogx.example.com/custom/v1"


class TestBaseUrlNormalization:
    def test_exits_when_path_missing(self, discovery: CodexServerDiscovery) -> None:
        with pytest.raises(SystemExit):
            discovery.normalize_base_url("https://ogx.example.com")

    def test_preserves_explicit_path(self, discovery: CodexServerDiscovery) -> None:
        assert discovery.normalize_base_url("https://ogx.example.com/prefix/v1") == "https://ogx.example.com/prefix/v1"

    def test_preserves_explicit_v1_path(self, discovery: CodexServerDiscovery) -> None:
        assert discovery.normalize_base_url("https://ogx.example.com/v1") == "https://ogx.example.com/v1"

    def test_exits_when_path_does_not_include_v1(self, discovery: CodexServerDiscovery) -> None:
        with pytest.raises(SystemExit):
            discovery.normalize_base_url("https://ogx.example.com/api")

    def test_exits_when_base_url_invalid(self, discovery: CodexServerDiscovery) -> None:
        with pytest.raises(SystemExit):
            discovery.normalize_base_url("localhost:8321")


class TestCodexDetection:
    def test_exits_when_codex_not_in_path(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args([])
        with patch("ogx.cli.connect.codex.shutil.which", return_value=None):
            with pytest.raises(SystemExit):
                connect_codex._run_connect_codex_cmd(args)

    def test_continues_when_codex_found(self, connect_codex: ConnectCodex) -> None:
        args = connect_codex.parser.parse_args(["--model", "openai/gpt-4o"])
        mock_client = _make_mock_client([_make_model("openai/gpt-4o")])

        with (
            patch("ogx.cli.connect.codex.shutil.which", return_value="/usr/bin/codex"),
            patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client),
            patch("ogx.cli.connect.codex.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit) as exc_info:
                connect_codex._run_connect_codex_cmd(args)
            assert exc_info.value.code == 0


class TestServerProbe:
    def test_uses_explicit_timeout(self, connect_codex: ConnectCodex, discovery: CodexServerDiscovery) -> None:
        mock_client = _make_mock_client([_make_model("openai/gpt-4o")])

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client) as mock_openai:
            discovery.fetch_models("http://localhost:8321/v1")

        assert mock_openai.call_args.kwargs["timeout"] == connect_codex.MODEL_DISCOVERY_TIMEOUT_SECONDS
        assert mock_openai.call_args.kwargs["api_key"] == "unused"
        assert mock_openai.call_args.kwargs["default_headers"] is None

    def test_forwards_auth_headers_to_probe(
        self, discovery: CodexServerDiscovery, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_client = _make_mock_client([_make_model("openai/gpt-4o")])
        monkeypatch.setenv("OGX_API_KEY", "secret-token")
        monkeypatch.setenv("OGX_PROVIDER_DATA", '{"passthrough_api_key":"abc"}')

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client) as mock_openai:
            discovery.fetch_models("https://ogx.example.com/v1")

        assert mock_openai.call_args.kwargs["api_key"] == "secret-token"
        assert mock_openai.call_args.kwargs["default_headers"] == {
            "X-OGX-Provider-Data": '{"passthrough_api_key":"abc"}'
        }

    def test_exits_when_server_unreachable(self, discovery: CodexServerDiscovery) -> None:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = APIConnectionError(request=httpx.Request("GET", "http://localhost"))

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                discovery.fetch_models("http://localhost:8321/v1")

    def test_exits_when_server_probe_times_out(self, discovery: CodexServerDiscovery) -> None:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = APITimeoutError(request=httpx.Request("GET", "http://localhost"))

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                discovery.fetch_models("http://localhost:8321/v1")

    def test_exits_on_server_error(self, discovery: CodexServerDiscovery) -> None:
        mock_client = MagicMock()
        mock_response = httpx.Response(500, request=httpx.Request("GET", "http://localhost"))
        mock_client.models.list.side_effect = APIStatusError("server error", response=mock_response, body=None)

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            with pytest.raises(SystemExit):
                discovery.fetch_models("http://localhost:8321/v1")

    def test_returns_models_on_success(self, discovery: CodexServerDiscovery) -> None:
        mock_client = _make_mock_client([_make_model("openai/gpt-4o"), _make_model("meta/llama-3.1-8b")])

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            models = discovery.fetch_models("http://localhost:8321/v1")
        assert [model.model_id for model in models] == ["openai/gpt-4o", "meta/llama-3.1-8b"]


class TestModelSelection:
    def test_uses_specified_model(self, discovery: CodexServerDiscovery) -> None:
        result = discovery.select_default_model(
            "openai/gpt-4o",
            [
                DiscoveredCodexModel("openai/gpt-4o", {}),
                DiscoveredCodexModel("meta/llama-3.1-8b", {}),
            ],
        )
        assert result.model_id == "openai/gpt-4o"

    def test_exits_when_specified_model_not_found(self, discovery: CodexServerDiscovery) -> None:
        with pytest.raises(SystemExit):
            discovery.select_default_model(
                "nonexistent",
                [
                    DiscoveredCodexModel("openai/gpt-4o", {}),
                    DiscoveredCodexModel("meta/llama-3.1-8b", {}),
                ],
            )

    def test_defaults_to_first_model(self, discovery: CodexServerDiscovery) -> None:
        result = discovery.select_default_model(
            None,
            [
                DiscoveredCodexModel("openai/gpt-4o", {}),
                DiscoveredCodexModel("meta/llama-3.1-8b", {}),
            ],
        )
        assert result.model_id == "openai/gpt-4o"

    def test_filters_out_embedding_models(self, discovery: CodexServerDiscovery) -> None:
        mock_client = _make_mock_client(
            [
                _make_model("openai/gpt-4o"),
                _make_model("openai/text-embedding-3-small", model_type="embedding"),
            ]
        )

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            models = discovery.fetch_models("http://localhost:8321/v1")
        assert [model.model_id for model in models] == ["openai/gpt-4o"]

    def test_exits_when_no_llm_models(self, discovery: CodexServerDiscovery) -> None:
        mock_client = _make_mock_client(
            [
                _make_model("openai/text-embedding-3-small", model_type="embedding"),
            ]
        )

        with patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client):
            models = discovery.fetch_models("http://localhost:8321/v1")
        assert models == []


class TestSessionConfigGeneration:
    def test_builds_interactive_codex_command(self, connect_codex: ConnectCodex) -> None:
        assert connect_codex._build_codex_command() == ["codex", "-p", "ogx"]

    def test_builds_exec_codex_command(self, connect_codex: ConnectCodex) -> None:
        assert connect_codex._build_codex_command("Reply with only Paris") == [
            "codex",
            "exec",
            "-p",
            "ogx",
            "Reply with only Paris",
        ]

    def test_builds_model_catalog_entry(self, catalog_builder: CodexCatalogBuilder) -> None:
        entry = catalog_builder.build_model_catalog_entry(
            DiscoveredCodexModel(
                "openai/gpt-4o",
                {
                    "description": "Primary OGX model",
                    "context_length": 256000,
                    "supported_reasoning_levels": [
                        {"effort": "medium", "description": "Balanced"},
                        {"effort": "high", "description": "Deep reasoning"},
                    ],
                    "default_reasoning_level": "medium",
                },
            ),
            index=0,
            is_default=True,
        )

        assert entry["slug"] == "openai/gpt-4o"
        assert entry["description"] == "Primary OGX model"
        assert entry["context_window"] == 256000
        assert entry["base_instructions"] == ""
        assert entry["supported_reasoning_levels"][0]["effort"] == "medium"
        assert entry["default_reasoning_level"] == "medium"
        assert entry["priority"] == 0

    def test_falls_back_when_zero_values_are_provided_as_strings(self, catalog_builder: CodexCatalogBuilder) -> None:
        entry = catalog_builder.build_model_catalog_entry(
            DiscoveredCodexModel(
                "openai/gpt-4o",
                {
                    "context_length": "0",
                    "auto_compact_token_limit": "0",
                },
            ),
            index=0,
            is_default=True,
        )

        assert entry["context_window"] == catalog_builder.DEFAULT_CONTEXT_WINDOW
        assert entry["max_context_window"] == catalog_builder.DEFAULT_CONTEXT_WINDOW
        assert entry["auto_compact_token_limit"] is None

    def test_writes_generated_config_and_model_catalog(self, session_builder: CodexSessionBuilder, tmp_path) -> None:
        models = [
            DiscoveredCodexModel(
                "openai/gpt-4o",
                {
                    "description": "Primary OGX model",
                    "context_length": 256000,
                    "supported_reasoning_levels": [
                        {"effort": "medium", "description": "Balanced"},
                        {"effort": "high", "description": "Deep reasoning"},
                    ],
                    "default_reasoning_level": "medium",
                },
            ),
            DiscoveredCodexModel("meta/llama-3.1-8b", {}),
        ]

        session_builder.write_session_files(tmp_path, "https://ogx.example.com/v1", models, "openai/gpt-4o")

        config = tomllib.loads((tmp_path / "ogx.config.toml").read_text())
        catalog = json.loads((tmp_path / "ogx-model-catalog.json").read_text())

        assert not (tmp_path / "config.toml").exists()
        assert config["model"] == "openai/gpt-4o"
        assert config["model_provider"] == "ogx"
        assert config["model_catalog_json"] == str(tmp_path / "ogx-model-catalog.json")
        assert config["features"]["multi_agent"] is False
        assert config["model_providers"]["ogx"]["base_url"] == "https://ogx.example.com/v1"
        assert "env_key" not in config["model_providers"]["ogx"]
        assert config["model_providers"]["ogx"]["env_http_headers"] == {"X-OGX-Provider-Data": "OGX_PROVIDER_DATA"}

        assert catalog["models"][0]["slug"] == "openai/gpt-4o"
        assert catalog["models"][0]["description"] == "Primary OGX model"
        assert catalog["models"][0]["context_window"] == 256000
        assert catalog["models"][0]["supported_reasoning_levels"][0]["effort"] == "medium"
        assert catalog["models"][0]["default_reasoning_level"] == "medium"
        assert catalog["models"][1]["display_name"] == "meta/llama-3.1-8b"
        assert catalog["models"][1]["description"] == "Model exposed by the running OGX server as meta/llama-3.1-8b."
        assert catalog["models"][1]["supported_reasoning_levels"] == []

    def test_writes_env_key_when_ogx_api_key_is_set(
        self, session_builder: CodexSessionBuilder, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OGX_API_KEY", "secret-token")
        session_builder.write_session_files(
            tmp_path,
            "https://ogx.example.com/v1",
            [DiscoveredCodexModel("openai/gpt-4o", {})],
            "openai/gpt-4o",
        )

        config = tomllib.loads((tmp_path / "ogx.config.toml").read_text())
        assert config["model_providers"]["ogx"]["env_key"] == "OGX_API_KEY"


class TestConnect:
    def test_forwards_auth_env_to_codex_process(
        self, connect_codex: ConnectCodex, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args = connect_codex.parser.parse_args(["--model", "openai/gpt-4o"])
        monkeypatch.setenv("OGX_API_KEY", "secret-token")
        monkeypatch.setenv("OGX_PROVIDER_DATA", '{"passthrough_api_key":"abc"}')
        mock_client = _make_mock_client([_make_model("openai/gpt-4o")])

        with (
            patch("ogx.cli.connect.codex.shutil.which", return_value="/usr/bin/codex"),
            patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client),
            patch(
                "ogx.cli.connect.codex.tempfile.TemporaryDirectory",
                return_value=contextlib.nullcontext(str(tmp_path)),
            ),
            patch("ogx.cli.connect.codex.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit):
                connect_codex._run_connect_codex_cmd(args)

        launched_env = mock_run.call_args.kwargs["env"]
        assert launched_env["CODEX_HOME"] == str(tmp_path)
        assert launched_env["OGX_API_KEY"] == "secret-token"
        assert launched_env["OGX_PROVIDER_DATA"] == '{"passthrough_api_key":"abc"}'

        config = tomllib.loads((tmp_path / "ogx.config.toml").read_text())
        assert config["model_providers"]["ogx"]["env_key"] == "OGX_API_KEY"
        assert config["model_providers"]["ogx"]["env_http_headers"] == {"X-OGX-Provider-Data": "OGX_PROVIDER_DATA"}

    def test_launches_codex_with_generated_session_home(self, connect_codex: ConnectCodex, tmp_path) -> None:
        args = connect_codex.parser.parse_args(["--model", "openai/gpt-4o"])
        mock_client = _make_mock_client(
            [
                _make_model(
                    "openai/gpt-4o",
                    metadata={"context_length": 200000, "description": "OGX default model"},
                )
            ]
        )

        with (
            patch("ogx.cli.connect.codex.shutil.which", return_value="/usr/bin/codex"),
            patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client),
            patch(
                "ogx.cli.connect.codex.tempfile.TemporaryDirectory",
                return_value=contextlib.nullcontext(str(tmp_path)),
            ),
            patch("ogx.cli.connect.codex.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit):
                connect_codex._run_connect_codex_cmd(args)

        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == ["codex", "-p", "ogx"]
        assert mock_run.call_args.kwargs["env"]["CODEX_HOME"] == str(tmp_path)

        config = tomllib.loads((tmp_path / "ogx.config.toml").read_text())
        catalog = json.loads((tmp_path / "ogx-model-catalog.json").read_text())
        assert config["model"] == "openai/gpt-4o"
        assert config["features"]["multi_agent"] is False
        assert config["model_providers"]["ogx"]["base_url"] == "http://localhost:8321/v1"
        assert catalog["models"][0]["context_window"] == 200000

    def test_launches_codex_exec_with_generated_session_home(self, connect_codex: ConnectCodex, tmp_path) -> None:
        args = connect_codex.parser.parse_args(["--model", "openai/gpt-4o", "--exec", "Reply with only Paris"])
        mock_client = _make_mock_client([_make_model("openai/gpt-4o")])

        with (
            patch("ogx.cli.connect.codex.shutil.which", return_value="/usr/bin/codex"),
            patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client),
            patch(
                "ogx.cli.connect.codex.tempfile.TemporaryDirectory",
                return_value=contextlib.nullcontext(str(tmp_path)),
            ),
            patch("ogx.cli.connect.codex.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            with pytest.raises(SystemExit) as exc_info:
                connect_codex._run_connect_codex_cmd(args)

        assert exc_info.value.code == 0
        assert mock_run.call_args.args[0] == ["codex", "exec", "-p", "ogx", "Reply with only Paris"]
        assert mock_run.call_args.kwargs["env"]["CODEX_HOME"] == str(tmp_path)

    def test_propagates_exec_exit_code(self, connect_codex: ConnectCodex, tmp_path) -> None:
        args = connect_codex.parser.parse_args(["--model", "openai/gpt-4o", "--exec", "Reply with only Paris"])
        mock_client = _make_mock_client([_make_model("openai/gpt-4o")])

        with (
            patch("ogx.cli.connect.codex.shutil.which", return_value="/usr/bin/codex"),
            patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client),
            patch(
                "ogx.cli.connect.codex.tempfile.TemporaryDirectory",
                return_value=contextlib.nullcontext(str(tmp_path)),
            ),
            patch("ogx.cli.connect.codex.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=42)
            with pytest.raises(SystemExit) as exc_info:
                connect_codex._run_connect_codex_cmd(args)
            assert exc_info.value.code == 42

    def test_propagates_exit_code(self, connect_codex: ConnectCodex, tmp_path) -> None:
        args = connect_codex.parser.parse_args(["--model", "openai/gpt-4o"])
        mock_client = _make_mock_client([_make_model("openai/gpt-4o")])

        with (
            patch("ogx.cli.connect.codex.shutil.which", return_value="/usr/bin/codex"),
            patch("ogx.cli.connect.codex.OpenAI", return_value=mock_client),
            patch(
                "ogx.cli.connect.codex.tempfile.TemporaryDirectory",
                return_value=contextlib.nullcontext(str(tmp_path)),
            ),
            patch("ogx.cli.connect.codex.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=42)
            with pytest.raises(SystemExit) as exc_info:
                connect_codex._run_connect_codex_cmd(args)
            assert exc_info.value.code == 42
