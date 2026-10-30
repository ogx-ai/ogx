# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import sys
from typing import NoReturn
from urllib.parse import urlsplit

from termcolor import cprint

from ogx.cli.stack.utils import print_subcommand_description
from ogx.cli.subcommand import Subcommand

from .claude import ConnectClaude
from .codex import ConnectCodex
from .opencode import ConnectOpenCode


def _exit_with_error(message: str) -> NoReturn:
    cprint(message, color="red", file=sys.stderr)
    raise SystemExit(1)


def default_base_url() -> str:
    """The fallback OGX server URL, resolved in one place for all subcommands.

    Precedence: OGX_BASE_URL (a full URL) > http://localhost:OGX_PORT/v1.
    """
    env_url = os.getenv("OGX_BASE_URL")
    if env_url:
        return env_url
    return f"http://localhost:{os.getenv('OGX_PORT', 8321)}/v1"


def validate_base_url(raw_base_url: str) -> str:
    """Validate an OGX server URL and return its canonical form.

    The URL must be a full OpenAI-compatible API base URL whose path ends in
    /v1; subcommands receive the canonical form and never re-derive it.
    """
    base_url = raw_base_url.strip().rstrip("/")
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        _exit_with_error(
            f"Failed to parse OGX base URL '{raw_base_url}'.\n"
            "Provide a full OpenAI-compatible API base URL such as "
            "http://localhost:8321/v1 or https://ogx.example.com/v1."
        )
    try:
        _ = parsed.port
    except ValueError:
        _exit_with_error(f"Failed to parse OGX base URL '{raw_base_url}'.\nThe URL port is not a valid number.")
    if not parsed.path.endswith("/v1"):
        _exit_with_error(
            f"Failed to parse OGX base URL '{raw_base_url}'.\nInclude the OpenAI-compatible API path, such as /v1."
        )
    return base_url


class ConnectParser(Subcommand):
    """Top-level CLI parser for the 'ogx connect' command group.

    `connect` is a command subgroup used to connect OGX to external services.
    """

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "connect",
            prog="ogx connect",
            description="Connect third-party tools to the running OGX server",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        self._help_only = self._print_help_only
        self.parser.set_defaults(func=self._help_only)
        self.parser.add_argument(
            "--url",
            dest="connect_url",
            type=str,
            default=None,
            help=(
                "Base URL of the OGX server, including /v1. Applies to all connect subcommands. "
                "If omitted, falls back to the OGX_BASE_URL env var, then to "
                "http://localhost:OGX_PORT/v1."
            ),
        )

        self.parser.add_argument(
            "--insecure",
            "--no-check-certificate",
            "-k",
            action="store_true",
            default=False,
            help=(
                "Allow HTTPS connections to servers with untrusted (for example, self-signed) "
                "certificates, like curl -k or wget --no-check-certificate. Only affects this "
                "connect invocation."
            ),
        )

        subparsers = self.parser.add_subparsers(title="connect_subcommands")

        ConnectClaude.create(subparsers)
        ConnectOpenCode.create(subparsers)
        ConnectCodex.create(subparsers)
        print_subcommand_description(self.parser, subparsers)

        self._install_url_resolution()

    def _install_url_resolution(self) -> None:
        # The root parser invokes the raw subparser's parse_known_args, not
        # any method on this wrapper, so hook the parser instance to resolve
        # the server URL exactly once per `ogx connect ...` invocation.
        # Subcommands read args.connect_url and never resolve it themselves.
        parser = self.parser
        original = parser.parse_known_args

        def parse_known_args(
            args: list[str] | None = None, namespace: argparse.Namespace | None = None
        ) -> tuple[argparse.Namespace, list[str]]:
            parsed, extras = original(args, namespace)
            if parsed.func is not self._help_only:
                if parsed.connect_url is None:
                    parsed.connect_url = default_base_url()
                parsed.connect_url = validate_base_url(parsed.connect_url)
            return parsed, extras

        parser.parse_known_args = parse_known_args  # type: ignore[method-assign]

    def _print_help_only(self, args: argparse.Namespace) -> None:
        self.parser.print_help()
