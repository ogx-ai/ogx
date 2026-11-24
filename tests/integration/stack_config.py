# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""The single parser for the integration suite's --stack-config value space.

Every entry point -- the pytest client fixture (fixtures/common.py), test gating
(conftest.py, telemetry/conftest.py) and scripts/integration-tests.sh (through the
``shell`` command below) -- classifies a value through parse_stack_config(), so the same
string always means the same thing everywhere.

Kept out of conftest.py so other modules can import it as a plain module instead of through
pytest's own conftest-loading machinery, which can give the same file two distinct module
objects depending on how it's reached -- silently breaking an `is`/set-membership check against
one of this module's constants if the checker imported a different copy than conftest.py did.
It also imports only the standard library, so the runner script can execute it directly.

Accepted values:

- ``<distro>`` / ``<path/to/config.yaml>`` / ``<distro>::<run_file.yaml>``: in-process library client
- ``<api>=<provider>,...``: dynamic spec, in-process library client
- ``server:<config>[:<port>]`` / ``server:<distro>::<run_file.yaml>``: a server spawned from <config>
- ``docker:<distro>``: a container built and started by scripts/integration-tests.sh only
- ``http(s)://host[:port]``: an already-running server
"""

import argparse
import os
import shlex
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_SERVER_PORT = 8321
RUNNER_SCRIPT = "scripts/integration-tests.sh"


class StackConfigKind(StrEnum):
    LIBRARY = "library"
    DYNAMIC = "dynamic"
    SERVER = "server"
    DOCKER = "docker"
    URL = "url"


@dataclass(frozen=True)
class StackConfig:
    """A parsed --stack-config value.

    target is what the kind operates on: the distro name / yaml path for LIBRARY and SERVER,
    the k=v spec for DYNAMIC, the distro for DOCKER and the base URL for URL. port is only set
    for SERVER, and only when the value carries an explicit ``:<port>`` suffix.
    """

    raw: str
    kind: StackConfigKind
    target: str
    port: int | None = None

    @property
    def is_server(self) -> bool:
        """Whether tests talk to a real server process rather than an in-process library client."""
        return self.kind in (StackConfigKind.SERVER, StackConfigKind.DOCKER, StackConfigKind.URL)

    def server_port(self, environ: Mapping[str, str] = os.environ) -> int:
        """The port a SERVER config listens on: the explicit suffix, else OGX_PORT, else the default."""
        if self.port is not None:
            return self.port
        return int(environ.get("OGX_PORT") or DEFAULT_SERVER_PORT)

    def deps_args(self) -> list[str] | None:
        """Arguments for `ogx stack list-deps` that resolve this config's provider
        dependencies, or None when dependencies are not installed locally (docker images bake
        them in; a URL points at someone else's server)."""
        if self.kind in (StackConfigKind.DOCKER, StackConfigKind.URL):
            return None
        if "=" in self.target:
            return ["--providers", self.target]
        return [self.target]


def parse_stack_config(value: str) -> StackConfig:
    """Classify a --stack-config / OGX_CONFIG value. Raises ValueError on malformed values."""
    value = value.strip()
    if not value:
        raise ValueError("Failed to parse stack config: value is empty")

    if value.startswith("server:"):
        rest = value.removeprefix("server:")
        if "::" in rest:
            # <distro>::<run_file.yaml>; a port can't be appended unambiguously, use OGX_PORT.
            return StackConfig(raw=value, kind=StackConfigKind.SERVER, target=rest)
        name, sep, port_str = rest.partition(":")
        if not name:
            raise ValueError(f"Failed to parse stack config '{value}': missing config name after 'server:'")
        port = None
        if sep:
            if not port_str.isdigit() or not 0 < int(port_str) < 65536:
                raise ValueError(f"Failed to parse stack config '{value}': '{port_str}' is not a valid port")
            port = int(port_str)
        return StackConfig(raw=value, kind=StackConfigKind.SERVER, target=name, port=port)

    if value.startswith("docker:"):
        distro = value.removeprefix("docker:")
        if not distro:
            raise ValueError(f"Failed to parse stack config '{value}': missing distro name after 'docker:'")
        return StackConfig(raw=value, kind=StackConfigKind.DOCKER, target=distro)

    parsed_url = urlparse(value)
    if parsed_url.scheme in ("http", "https") and parsed_url.netloc:
        return StackConfig(raw=value, kind=StackConfigKind.URL, target=value)

    if "=" in value:
        return StackConfig(raw=value, kind=StackConfigKind.DYNAMIC, target=value)

    return StackConfig(raw=value, kind=StackConfigKind.LIBRARY, target=value)


def resolve_stack_config(option_value: str | None, environ: Mapping[str, str] = os.environ) -> StackConfig | None:
    """The effective stack config: --stack-config if given, else OGX_CONFIG, else None.

    Every consumer goes through this, so an OGX_CONFIG-only run is classified (and gated)
    exactly like the equivalent --stack-config run.
    """
    value = option_value or environ.get("OGX_CONFIG")
    if not value:
        return None
    return parse_stack_config(value)


def get_stack_config(config: Any) -> StackConfig | None:
    """resolve_stack_config() for a pytest.Config (typed Any to keep this module pytest-free)."""
    return resolve_stack_config(config.getoption("--stack-config", default=None))


def is_server_stack_config(stack_config: str | None) -> bool:
    """Whether a --stack-config value points at a real server process (server:, docker:, or
    an http(s) URL) rather than an in-process library client."""
    if not stack_config:
        return False
    try:
        return parse_stack_config(stack_config).is_server
    except ValueError:
        return False


# Environment shared by the server process and the pytest process in server mode. The runner
# script exports these before starting its server, and the pytest fixture applies the same
# defaults before spawning one, so server-mode-only tests see the same environment either way.
SERVER_MODE_ENV_DEFAULTS: dict[str, str] = {
    # Fixed collector port so the server can connect to the telemetry test collector.
    "OGX_TEST_COLLECTOR_PORT": "4317",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    "OTEL_BSP_SCHEDULE_DELAY": "200",
    "OTEL_BSP_EXPORT_TIMEOUT": "2000",
    "OTEL_METRIC_EXPORT_INTERVAL": "200",
    # Standalone metrics scrape server, scraped by tests/integration/inspect/test_metrics_endpoint.py.
    "OGX_METRICS_ENDPOINT_ENABLED": "1",
}


def apply_server_mode_env_defaults(environ: MutableMapping[str, str] = os.environ) -> None:
    for key, value in SERVER_MODE_ENV_DEFAULTS.items():
        environ.setdefault(key, value)


# Paths (relative to the repo root) whose gating needs the parsed stack config, unavailable to
# a module-level `pytestmark = pytest.mark.skipif(...)`. Skipped by conftest.py's
# pytest_collection_modifyitems instead, once the config is available; see the modules
# themselves for why each is gated. Stored as Path.parts rather than a plain string so the
# match isn't sensitive to the OS path separator (str(Path(...)) renders backslashes on
# Windows).
SERVER_ONLY_TEST_PATHS = {Path("tests/integration/inspect/test_metrics_endpoint.py").parts}
LIBRARY_CLIENT_ONLY_TEST_PATHS = {Path("tests/integration/inference/test_inference_store_disabled.py").parts}


def _shell_assignments(stack_config: StackConfig) -> str:
    """Bash assignments describing stack_config, for `eval` in scripts/integration-tests.sh."""
    deps_args = stack_config.deps_args()
    lines = [
        f"STACK_CONFIG_KIND={shlex.quote(stack_config.kind.value)}",
        f"STACK_CONFIG_TARGET={shlex.quote(stack_config.target)}",
        f"STACK_CONFIG_PORT={shlex.quote(str(stack_config.port) if stack_config.port is not None else '')}",
        f"STACK_CONFIG_IS_SERVER={'true' if stack_config.is_server else 'false'}",
        "STACK_CONFIG_DEPS_ARGS=(" + " ".join(shlex.quote(arg) for arg in deps_args or []) + ")",
    ]
    return "\n".join(lines)


def _server_env_exports() -> str:
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in SERVER_MODE_ENV_DEFAULTS.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse an integration-test --stack-config value.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    shell_parser = subparsers.add_parser("shell", help="Print bash assignments describing the value")
    shell_parser.add_argument("value")
    subparsers.add_parser("server-env", help="Print bash exports for the server-mode environment")
    args = parser.parse_args(argv)

    if args.command == "server-env":
        print(_server_env_exports())
        return 0
    try:
        print(_shell_assignments(parse_stack_config(args.value)))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
