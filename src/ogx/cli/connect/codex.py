# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlsplit, urlunsplit

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from termcolor import cprint

from ogx.cli.subcommand import Subcommand
from ogx.log import get_logger

logger = get_logger(name=__name__, category="cli")


def _exit_with_error(message: str) -> NoReturn:
    cprint(message, color="red", file=sys.stderr)
    raise SystemExit(1)


@dataclass(frozen=True)
class DiscoveredCodexModel:
    """Model entry returned by OGX and adapted into the generated Codex catalog."""

    model_id: str
    custom_metadata: dict[str, Any]


class CodexServerDiscovery:
    """Probe the OGX server and normalize its model list for Codex."""

    def __init__(self, *, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    def normalize_base_url(self, raw_base_url: str) -> str:
        base_url = raw_base_url.strip()
        parsed = urlsplit(base_url)
        if not parsed.scheme or not parsed.netloc:
            _exit_with_error(
                f"Failed to parse OGX base URL '{raw_base_url}'.\n"
                "Provide a full OpenAI-compatible API base URL such as "
                "http://localhost:8321/v1 or https://ogx.example.com/v1."
            )
        if not parsed.path.rstrip("/").endswith("/v1"):
            _exit_with_error(
                f"Failed to parse OGX base URL '{raw_base_url}'.\nInclude the OpenAI-compatible API path, such as /v1."
            )

        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))

    def fetch_models(self, base_url: str) -> list[DiscoveredCodexModel]:
        default_headers = self.build_request_headers()
        client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("OGX_API_KEY", "unused"),
            timeout=self.timeout_seconds,
            default_headers=default_headers or None,
        )
        try:
            response = client.models.list()
        except APITimeoutError:
            _exit_with_error(
                f"Failed to connect to OGX server at {base_url}\n"
                f"Timed out while querying available models after {self.timeout_seconds} seconds."
            )
        except APIConnectionError:
            _exit_with_error(
                f"Failed to connect to OGX server at {base_url}\nStart the server first with: ogx run <config>"
            )
        except APIStatusError as e:
            _exit_with_error(f"Failed to query models from OGX server at {base_url} (HTTP {e.status_code})")

        models: list[DiscoveredCodexModel] = []
        for model in response.data:
            metadata = self.extract_custom_metadata(model)
            if metadata.get("model_type") != "embedding":
                models.append(DiscoveredCodexModel(model_id=model.id, custom_metadata=metadata))
        return models

    @staticmethod
    def extract_custom_metadata(model: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        model_extra = getattr(model, "model_extra", None)
        if isinstance(model_extra, dict):
            extra_custom_metadata = model_extra.get("custom_metadata")
            if isinstance(extra_custom_metadata, dict):
                metadata.update(extra_custom_metadata)

        direct_custom_metadata = getattr(model, "custom_metadata", None)
        if isinstance(direct_custom_metadata, dict):
            metadata.update(direct_custom_metadata)

        return metadata

    @staticmethod
    def build_request_headers() -> dict[str, str]:
        headers: dict[str, str] = {}
        provider_data = os.getenv("OGX_PROVIDER_DATA", "").strip()
        if provider_data:
            headers["X-OGX-Provider-Data"] = provider_data
        return headers

    @staticmethod
    def select_default_model(
        requested_model: str | None, available_models: list[DiscoveredCodexModel]
    ) -> DiscoveredCodexModel:
        available_model_ids = [model.model_id for model in available_models]
        if requested_model:
            if requested_model not in available_model_ids:
                _exit_with_error(
                    f"Failed to find model '{requested_model}' on the OGX server.\n"
                    f"Available models: {', '.join(available_model_ids)}"
                )
            return next(model for model in available_models if model.model_id == requested_model)

        return available_models[0]


class CodexCatalogBuilder:
    """Translate discovered OGX models into the generated Codex catalog schema."""

    DEFAULT_CONTEXT_WINDOW = 128000

    def __init__(
        self,
        *,
        default_context_window: int = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        self.default_context_window = default_context_window

    def build_model_catalog(
        self, available_models: list[DiscoveredCodexModel], default_model: str
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "models": [
                self.build_model_catalog_entry(model, index=index, is_default=model.model_id == default_model)
                for index, model in enumerate(available_models)
            ]
        }

    def build_model_catalog_entry(self, model: DiscoveredCodexModel, *, index: int, is_default: bool) -> dict[str, Any]:
        metadata = model.custom_metadata
        context_window = self._coerce_int(
            metadata.get("context_window") or metadata.get("context_length"),
            self.default_context_window,
        )
        entry: dict[str, Any] = {
            "slug": model.model_id,
            "display_name": self._coerce_str(
                metadata.get("display_name") or metadata.get("provider_model_id"),
                model.model_id,
            ),
            "description": self._coerce_str(
                metadata.get("description"),
                f"Model exposed by the running OGX server as {model.model_id}.",
            ),
            "default_reasoning_level": None,
            "context_window": context_window,
            "max_context_window": context_window,
            "auto_compact_token_limit": self._coerce_optional_int(metadata.get("auto_compact_token_limit")),
            "shell_type": "default",
            "additional_speed_tiers": [],
            "service_tiers": [],
            "default_service_tier": None,
            "availability_nux": None,
            "upgrade": None,
            "base_instructions": "",
            "model_messages": None,
            "supports_reasoning_summaries": False,
            "default_reasoning_summary": "auto",
            "support_verbosity": False,
            "default_verbosity": None,
            "apply_patch_tool_type": None,
            "web_search_tool_type": "text",
            "truncation_policy": {"mode": "bytes", "limit": 10000},
            "supports_parallel_tool_calls": False,
            "supports_image_detail_original": False,
            "effective_context_window_percent": 95,
            "experimental_supported_tools": [],
            "input_modalities": self._coerce_string_list(metadata.get("input_modalities"), fallback=["text"]),
            "supported_reasoning_levels": [],
            "used_fallback_model_metadata": False,
            "supports_search_tool": False,
            "visibility": "list",
            "priority": 0 if is_default else index + 1,
            "supported_in_api": True,
        }

        supported_reasoning_levels = self.build_reasoning_levels(metadata)
        if supported_reasoning_levels:
            entry["supported_reasoning_levels"] = supported_reasoning_levels
            entry["default_reasoning_level"] = self._coerce_str(
                metadata.get("default_reasoning_level") or metadata.get("defaultReasoningEffort"),
                supported_reasoning_levels[0]["effort"],
            )

        return entry

    @staticmethod
    def build_reasoning_levels(metadata: dict[str, Any]) -> list[dict[str, str]]:
        raw_levels = metadata.get("supported_reasoning_levels") or metadata.get("supportedReasoningEfforts") or []
        if not isinstance(raw_levels, list):
            return []

        levels: list[dict[str, str]] = []
        for item in raw_levels:
            if not isinstance(item, dict):
                continue
            effort = item.get("effort") or item.get("reasoningEffort")
            description = item.get("description")
            if isinstance(effort, str) and isinstance(description, str):
                levels.append({"effort": effort, "description": description})
        return levels

    @staticmethod
    def _coerce_int(value: Any, fallback: int) -> int:
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit():
            parsed = int(value)
            if parsed > 0:
                return parsed
        return fallback

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit():
            parsed = int(value)
            if parsed > 0:
                return parsed
        return None

    @staticmethod
    def _coerce_str(value: Any, fallback: str) -> str:
        if isinstance(value, str) and value.strip():
            return value
        return fallback

    @staticmethod
    def _coerce_string_list(value: Any, fallback: list[str]) -> list[str]:
        if isinstance(value, list):
            items = [item for item in value if isinstance(item, str) and item]
            if items:
                return items
        return [*fallback]


class CodexSessionBuilder:
    """Render the generated Codex session files for an OGX-backed profile."""

    def __init__(self, *, catalog_builder: CodexCatalogBuilder | None = None) -> None:
        self.catalog_builder = catalog_builder or CodexCatalogBuilder()

    def write_session_files(
        self,
        codex_home: Path,
        base_url: str,
        available_models: list[DiscoveredCodexModel],
        default_model: str,
    ) -> None:
        model_catalog_path = codex_home / "ogx-model-catalog.json"
        config_path = codex_home / "ogx.config.toml"
        model_catalog_path.write_text(
            json.dumps(self.catalog_builder.build_model_catalog(available_models, default_model), indent=2)
        )
        config_path.write_text(self.build_codex_config(base_url, model_catalog_path, default_model))

    @staticmethod
    def build_codex_config(base_url: str, model_catalog_path: Path, default_model: str) -> str:
        env_http_headers = '{ "X-OGX-Provider-Data" = "OGX_PROVIDER_DATA" }'
        config_lines = [
            f"model = {json.dumps(default_model)}",
            'model_provider = "ogx"',
            f"model_catalog_json = {json.dumps(str(model_catalog_path))}",
            "",
            "[features]",
            "multi_agent = false",
            "",
            "[model_providers.ogx]",
            'name = "OGX"',
            f"base_url = {json.dumps(base_url)}",
            'wire_api = "responses"',
            "supports_websockets = false",
        ]
        if os.getenv("OGX_API_KEY", "").strip():
            config_lines.extend(
                [
                    'env_key = "OGX_API_KEY"',
                    'env_key_instructions = "Set OGX_API_KEY when your OGX deployment requires bearer authentication."',
                ]
            )
        config_lines.extend(
            [
                f"env_http_headers = {env_http_headers}",
                "",
            ]
        )
        return "\n".join(config_lines)


class ConnectCodex(Subcommand):
    """Connect Codex to the running OGX server."""

    MODEL_DISCOVERY_TIMEOUT_SECONDS = 20
    DEFAULT_BASE_URL = "http://localhost:8321/v1"

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.discovery = CodexServerDiscovery(timeout_seconds=self.MODEL_DISCOVERY_TIMEOUT_SECONDS)
        self.session_builder = CodexSessionBuilder()
        self.parser = subparsers.add_parser(
            "codex",
            prog="ogx connect codex",
            description="Launch Codex connected to the running OGX server.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_connect_codex_cmd)

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Default model ID. If omitted, the first available model is used.",
        )
        self.parser.add_argument(
            "--url",
            type=str,
            default=os.getenv("OGX_BASE_URL", self.DEFAULT_BASE_URL),
            help="OGX OpenAI-compatible API base URL, including /v1.",
        )
        self.parser.add_argument(
            "--exec",
            dest="exec_prompt",
            type=str,
            default=None,
            help="Run Codex non-interactively with the provided prompt.",
        )

    def _run_connect_codex_cmd(self, args: argparse.Namespace) -> None:
        if not shutil.which("codex"):
            _exit_with_error("Failed to find 'codex' in PATH. Install it from https://github.com/openai/codex")

        base_url = self.discovery.normalize_base_url(args.url)

        models = self.discovery.fetch_models(base_url)
        if not models:
            _exit_with_error("Failed to find any LLM models on the OGX server.")

        default_model = self.discovery.select_default_model(args.model, models)

        logger.info("Connecting to Codex", default_model=default_model.model_id, models=len(models), base_url=base_url)

        with tempfile.TemporaryDirectory(prefix="ogx-codex-") as codex_home:
            codex_home_path = Path(codex_home)
            self.session_builder.write_session_files(codex_home_path, base_url, models, default_model.model_id)
            command = self._build_codex_command(args.exec_prompt)
            env = {**os.environ, "CODEX_HOME": str(codex_home_path)}
            result = subprocess.run(command, env=env)
            sys.exit(result.returncode)

    def _build_codex_command(self, exec_prompt: str | None = None) -> list[str]:
        if exec_prompt:
            return ["codex", "exec", "-p", "ogx", exec_prompt]
        return ["codex", "-p", "ogx"]
