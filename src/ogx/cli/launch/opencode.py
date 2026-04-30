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

from openai import APIConnectionError, APIStatusError, OpenAI
from termcolor import cprint

from ogx.cli.subcommand import Subcommand
from ogx.log import get_logger

logger = get_logger(name=__name__, category="cli")


class LaunchOpenCode(Subcommand):
    """Launch OpenCode configured to use the running OGX server."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "opencode",
            prog="ogx launch opencode",
            description="Launch OpenCode connected to the running OGX server.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_launch_opencode_cmd)

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Default model ID. If omitted, the first available model is used.",
        )
        self.parser.add_argument(
            "--port",
            type=int,
            help="OGX server port.",
            default=int(os.getenv("OGX_PORT", 8321)),
        )
        self.parser.add_argument(
            "--host",
            type=str,
            default="localhost",
            help="OGX server host.",
        )

    def _run_launch_opencode_cmd(self, args: argparse.Namespace) -> None:
        if not shutil.which("opencode"):
            cprint(
                "Failed to find 'opencode' in PATH. Install it from https://opencode.ai",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)

        base_url = f"http://{args.host}:{args.port}/v1"

        models = self._fetch_models(base_url)
        if not models:
            cprint("Failed to find any LLM models on the OGX server.", color="red", file=sys.stderr)
            sys.exit(1)

        default_model = self._select_default_model(args.model, models)

        config = self._build_opencode_config(base_url, models, default_model)
        config_json = json.dumps(config)

        logger.info("Launching OpenCode", default_model=default_model, models=len(models), base_url=base_url)

        env = {**os.environ, "OPENCODE_CONFIG_CONTENT": config_json}
        result = subprocess.run(["opencode"], env=env)
        sys.exit(result.returncode)

    def _fetch_models(self, base_url: str) -> list[str]:
        client = OpenAI(base_url=base_url, api_key="unused")
        try:
            response = client.models.list()
        except APIConnectionError:
            cprint(
                f"Failed to connect to OGX server at {base_url}\nStart the server first with: ogx stack run <config>",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)
        except APIStatusError as e:
            cprint(
                f"Failed to query models from OGX server at {base_url} (HTTP {e.status_code})",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)

        models: list[str] = []
        for model in response.data:
            metadata = (model.model_extra or {}).get("custom_metadata") or {}
            if metadata.get("model_type") != "embedding":
                models.append(model.id)
        return models

    def _select_default_model(self, requested_model: str | None, available_models: list[str]) -> str:
        if requested_model:
            if requested_model not in available_models:
                cprint(
                    f"Failed to find model '{requested_model}' on the OGX server.\n"
                    f"Available models: {', '.join(available_models)}",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)
            return requested_model

        return available_models[0]

    def _build_opencode_config(self, base_url: str, all_models: list[str], default_model: str) -> dict:
        models_config = {
            model_id: {
                "name": model_id,
                "tools": True,
                "limit": {
                    "context": 128000,
                    "output": 4096,
                },
            }
            for model_id in all_models
        }
        return {
            "$schema": "https://opencode.ai/config.json",
            "model": f"ogx/{default_model}",
            "provider": {
                "ogx": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "OGX",
                    "options": {
                        "baseURL": base_url,
                    },
                    "models": models_config,
                }
            },
        }
