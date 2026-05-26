# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import shutil
import subprocess
import sys

from openai import APIConnectionError, APIStatusError, OpenAI
from termcolor import cprint

from ogx.cli.subcommand import Subcommand
from ogx.log import get_logger

logger = get_logger(name=__name__, category="cli")

_VARS_TO_UNSET = [
    "CLAUDE_CODE_USE_VERTEX",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "CLAUDE_CODE_USE_BEDROCK",
    "ANTHROPIC_BEDROCK_SESSION_TOKEN",
]


class ConnectClaude(Subcommand):
    """Connect Claude Code to the running OGX server."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "claude",
            prog="ogx connect claude",
            description="Launch Claude Code connected to the running OGX server.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_connect_claude_cmd)

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Model ID to map to all Claude tiers (haiku/sonnet/opus). If omitted, tiers are auto-detected from available models.",
        )
        self.parser.add_argument(
            "--haiku-model",
            type=str,
            default=None,
            help="Model ID for the haiku (fast) tier. Overrides --model for this tier.",
        )
        self.parser.add_argument(
            "--sonnet-model",
            type=str,
            default=None,
            help="Model ID for the sonnet (balanced) tier. Overrides --model for this tier.",
        )
        self.parser.add_argument(
            "--opus-model",
            type=str,
            default=None,
            help="Model ID for the opus (capable) tier. Overrides --model for this tier.",
        )
        default_port = os.getenv("OGX_PORT", "8321")
        self.parser.add_argument(
            "--url",
            type=str,
            default=f"http://localhost:{default_port}",
            help="OGX server URL.",
        )
        self.parser.add_argument(
            "--print-env",
            action="store_true",
            default=False,
            help="Print environment variables as shell export statements instead of launching Claude Code.",
        )
        self.parser.add_argument(
            "claude_args",
            nargs=argparse.REMAINDER,
            help="Additional arguments forwarded to the claude command (place after --).",
        )

    def _run_connect_claude_cmd(self, args: argparse.Namespace) -> None:
        if not args.print_env and not shutil.which("claude"):
            cprint(
                "Failed to find 'claude' in PATH. Install it from https://claude.com/download",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)

        base_url = args.url.rstrip("/")

        models = self._fetch_models(base_url)
        if not models:
            cprint("Failed to find any LLM models on the OGX server.", color="red", file=sys.stderr)
            sys.exit(1)

        model_mapping = self._resolve_model_mapping(
            model=args.model,
            haiku_model=args.haiku_model,
            sonnet_model=args.sonnet_model,
            opus_model=args.opus_model,
            available_models=models,
        )

        env = self._build_env(base_url, model_mapping)

        if args.print_env:
            for key in ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"]:
                print(f"export {key}={env[key]}")
            for key in [
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
            ]:
                if key in env:
                    print(f"export {key}={env[key]}")
            for key in _VARS_TO_UNSET:
                print(f"unset {key}")
            return

        claude_args = _strip_leading_separator(args.claude_args)

        logger.info(
            "Connecting to Claude Code",
            haiku=model_mapping.get("ANTHROPIC_DEFAULT_HAIKU_MODEL"),
            sonnet=model_mapping.get("ANTHROPIC_DEFAULT_SONNET_MODEL"),
            opus=model_mapping.get("ANTHROPIC_DEFAULT_OPUS_MODEL"),
            models=len(models),
            base_url=base_url,
        )

        result = subprocess.run(["claude", *claude_args], env=env)
        sys.exit(result.returncode)

    def _fetch_models(self, base_url: str) -> list[str]:
        client = OpenAI(base_url=base_url, api_key="unused")
        try:
            response = client.models.list()
        except APIConnectionError:
            cprint(
                f"Failed to connect to OGX server at {base_url}\nStart the server first with: ogx run <config>",
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

    def _resolve_model_mapping(
        self,
        model: str | None,
        haiku_model: str | None,
        sonnet_model: str | None,
        opus_model: str | None,
        available_models: list[str],
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        auto_detected = _detect_tier_models(available_models)
        tiers = [
            ("ANTHROPIC_DEFAULT_HAIKU_MODEL", haiku_model, "haiku"),
            ("ANTHROPIC_DEFAULT_SONNET_MODEL", sonnet_model, "sonnet"),
            ("ANTHROPIC_DEFAULT_OPUS_MODEL", opus_model, "opus"),
        ]

        for env_var, tier_model, tier_name in tiers:
            resolved = tier_model or model or auto_detected.get(tier_name) or available_models[0]
            if resolved not in available_models:
                cprint(
                    f"Failed to find model '{resolved}' on the OGX server.\n"
                    f"Available models: {', '.join(available_models)}",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)
            mapping[env_var] = resolved

        return mapping

    def _build_env(self, base_url: str, model_mapping: dict[str, str]) -> dict[str, str]:
        env = {**os.environ}
        env["ANTHROPIC_BASE_URL"] = base_url
        env["ANTHROPIC_AUTH_TOKEN"] = "ogx"  # noqa: S105 — placeholder, not a real secret
        env.update(model_mapping)
        for key in _VARS_TO_UNSET:
            env.pop(key, None)
        return env


def _detect_tier_models(available_models: list[str]) -> dict[str, str]:
    """Match available models to Claude tiers by looking for tier keywords in model names."""
    detected: dict[str, str] = {}
    for tier in ("haiku", "sonnet", "opus"):
        matches = [m for m in available_models if tier in m.lower()]
        if matches:
            detected[tier] = matches[0]
    return detected


def _strip_leading_separator(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args
