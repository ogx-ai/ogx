# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import enum
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

import httpx
import yaml
from termcolor import cprint

from ogx.cli.stack.run import _start_ui_development_server, _uvicorn_run
from ogx.cli.subcommand import Subcommand
from ogx.core.build import get_provider_dependencies
from ogx.core.stack import run_config_from_dynamic_config_spec
from ogx.core.utils.config_dirs import DISTRIBS_BASE_DIR
from ogx.log import get_logger
from ogx_api.models.models import ModelInput

logger = get_logger(name=__name__, category="cli")

# Model IDs that Claude Code looks for.
_CLAUDE_CODE_ALIASES: list[str] = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

# Inference provider IDs checked in priority order when building Claude Code aliases.
# Anthropic is preferred because the alias model IDs are native Anthropic identifiers;
# the others use provider_model_id="auto" to pick whatever model is available.
_CLAUDE_CODE_PROVIDER_PRIORITY: list[str] = ["anthropic", "ollama", "vllm", "openai"]


def _build_claude_code_aliases(providers_spec: str) -> list[ModelInput]:
    """Return ModelInput entries for Claude Code compatibility.

    Picks the highest-priority active inference provider from
    _CLAUDE_CODE_PROVIDER_PRIORITY and registers each alias in
    _CLAUDE_CODE_ALIASES against it. Anthropic providers receive a direct
    provider_model_id match; all others use "auto" to pick the first
    available LLM. Returns an empty list when no priority provider is active.
    """
    active_inference = {
        p.split("::", 1)[-1]
        for p in providers_spec.split(",")
        if p.startswith("inference=")
        for p in [p.split("=", 1)[1]]
    }

    chosen: str | None = None
    for candidate in _CLAUDE_CODE_PROVIDER_PRIORITY:
        if candidate in active_inference:
            chosen = candidate
            break

    if chosen is None:
        return []

    return [
        ModelInput(
            model_id=alias,
            provider_id=chosen,
            provider_model_id=alias if chosen == "anthropic" else "auto",
            metadata={"_unprefixed_alias": True},
        )
        for alias in _CLAUDE_CODE_ALIASES
    ]


class _ProbeStatus(enum.Enum):
    OK = "ok"
    NO_KEY = "no_key"
    AUTH = "auth"
    UNREACHABLE = "unreachable"


def add_letsgo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the server on. It can also be passed via the env var OGX_PORT.",
        default=int(os.getenv("OGX_PORT", 8321)),
    )
    parser.add_argument(
        "--enable-ui",
        action="store_true",
        help="Start the UI server",
    )
    parser.add_argument(
        "--persist-config",
        action="store_true",
        help="Persist generated runtime config to the distro directory",
    )
    parser.add_argument(
        "--providers-override",
        type=str,
        default=None,
        help="Explicit providers spec to use instead of auto-detection (e.g. inference=remote::ollama)",
    )
    parser.add_argument(
        "--skip-install-deps",
        action="store_true",
        help="Skip automatic installation of provider pip dependencies before starting the server.",
    )


def run_letsgo_cmd(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.enable_ui:
        try:
            _start_ui_development_server(args.port)
        except Exception:
            logger.warning("Failed to start UI development server", exc_info=True)

    if args.providers_override:
        providers_spec = args.providers_override
    else:
        providers_spec = _autodetect_providers()

    has_inference = any(p.startswith("inference=") for p in (providers_spec or "").split(","))
    if not has_inference:
        parser.error("No inference providers detected. Nothing to run.")

    distro_dir = DISTRIBS_BASE_DIR / "letsgo-run" if args.persist_config else Path(tempfile.mkdtemp())
    os.makedirs(distro_dir, exist_ok=True)

    try:
        run_config = run_config_from_dynamic_config_spec(
            dynamic_config_spec=providers_spec,
            distro_dir=distro_dir,
            distro_name="letsgo-run",
        )
    except ValueError as e:
        cprint(str(e), color="red", file=sys.stderr)
        sys.exit(1)

    if not args.skip_install_deps:
        normal_deps, special_deps, _ = get_provider_dependencies(run_config)
        _install_provider_deps(normal_deps, special_deps)

    claude_aliases = _build_claude_code_aliases(providers_spec)
    if claude_aliases:
        run_config.registered_resources.models.extend(claude_aliases)
        cprint(f"  ✓ Claude Code aliases → {claude_aliases[0].provider_id}", color="green")

    config_dict = run_config.model_dump(mode="json")

    config_file = distro_dir / "config.yaml"
    logger.info("Writing generated config to", config_file=config_file)
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    try:
        stack_args = argparse.Namespace()
        stack_args.port = args.port
        stack_args.enable_ui = args.enable_ui
        stack_args.providers = None
        _uvicorn_run(config_file, stack_args, parser)
    except Exception:
        logger.exception("Failed to start the stack server")
        raise


def _install_provider_deps(normal_deps: list[str], special_deps: list[str]) -> None:
    """Install provider pip dependencies into the current environment.

    Uses `uv pip install` when uv is available, falling back to `pip install`.
    A non-zero exit is logged as a warning rather than aborting startup,
    since packages may already satisfy the declared constraints.
    """
    if shutil.which("uv"):
        installer = ["uv", "pip", "install"]
    else:
        installer = [sys.executable, "-m", "pip", "install"]

    if normal_deps:
        cprint("Installing provider dependencies...", color="cyan")
        result = subprocess.run([*installer, *normal_deps])
        if result.returncode != 0:
            logger.warning("Failed to install provider dependencies", returncode=result.returncode)

    for special_dep in special_deps:
        result = subprocess.run([*installer, *special_dep.split()])
        if result.returncode != 0:
            logger.warning(
                "Failed to install special provider dependency", dep=special_dep, returncode=result.returncode
            )


def _autodetect_providers() -> str:
    """Probe all candidate providers and return a comma-separated providers spec string.

    Each provider is probed independently; all that pass are included in the
    result. Providers that require an API key skip the network probe entirely
    when the key environment variable is not set.
    """
    candidates = [
        # provider_type, env_for_base_url, default_base_url, probe_path, requires_api_key, api_key_env, extra_headers, api_key_header
        ("remote::ollama", "OLLAMA_URL", "http://localhost:11434/v1", "models", False, None, {}, None),
        ("remote::vllm", "VLLM_URL", "http://localhost:8000/v1", "health", False, None, {}, None),
        (
            "remote::llama-cpp-server",
            "LLAMA_CPP_SERVER_URL",
            "http://localhost:8080/v1",
            "models",
            False,
            None,
            {},
            None,
        ),
        ("remote::openai", "OPENAI_BASE_URL", "https://api.openai.com/v1", "models", True, "OPENAI_API_KEY", {}, None),
        (
            "remote::llama-openai-compat",
            "LLAMA_API_BASE_URL",
            "https://api.llama.com/compat/v1/",
            "models",
            True,
            "LLAMA_API_KEY",
            {},
            None,
        ),
        (
            "remote::anthropic",
            None,
            "https://api.anthropic.com/v1",
            "models",
            True,
            "ANTHROPIC_API_KEY",
            {"anthropic-version": "2023-06-01"},
            "x-api-key",
        ),
        (
            "remote::gemini",
            None,
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "models",
            True,
            "GEMINI_API_KEY",
            {},
            None,
        ),
        (
            "remote::azure",
            "AZURE_API_BASE",
            "",
            "openai/models?api-version=2024-12-01-preview",
            True,
            "AZURE_API_KEY",
            {},
            "api-key",
        ),
    ]

    passed: list[str] = []
    cprint("Scanning for available providers...", color="cyan")
    for (
        provider_type,
        base_env,
        default_base,
        probe_path,
        requires_key,
        key_env,
        extra_headers,
        api_key_header,
    ) in candidates:
        env_val: str | None = os.getenv(base_env) if base_env else None
        if env_val:
            base = env_val
            base_source = f"from {base_env}"
        else:
            base = default_base
            base_source = "default"

        status = _probe_endpoint(base, probe_path, requires_key, key_env, extra_headers, api_key_header)

        # Build annotation parts
        parts = [f"{base}, {base_source}"]
        if requires_key and key_env:
            parts.append(f"{key_env} {'set' if os.getenv(key_env) else 'not set'}")

        annotation = ", ".join(parts)

        if status == _ProbeStatus.OK:
            passed.append(f"inference={provider_type}")
            cprint(f"  ✓ {provider_type} ({annotation})", color="green")
        elif status == _ProbeStatus.NO_KEY:
            cprint(f"  ✗ {provider_type} ({annotation})", color="yellow")
        elif status == _ProbeStatus.AUTH:
            cprint(f"  ✗ {provider_type} ({annotation}) — auth error", color="yellow")
        else:
            cprint(f"  ✗ {provider_type} ({annotation}) — unreachable", color="yellow")

    # Inline providers require no external service — always include them.
    inline_providers = [
        "files=inline::localfs",
        "vector_io=inline::faiss",
        "tool_runtime=inline::file-search",
        "file_processors=inline::auto",
        "responses=inline::builtin",
        "messages=inline::builtin",
    ]
    cprint("  ✓ inline::localfs (built-in)", color="green")
    cprint("  ✓ inline::faiss (built-in)", color="green")
    cprint("  ✓ inline::file-search (built-in)", color="green")
    cprint("  ✓ inline::auto (built-in)", color="green")
    cprint("  ✓ inline::builtin responses (built-in)", color="green")
    cprint("  ✓ inline::builtin messages (built-in)", color="green")

    if passed:
        cprint(f"\nDetected {len(passed)} inference provider(s). Starting stack...", color="cyan")
    else:
        cprint("\nDetected no inference providers, not starting stack.", color="red")
    return ",".join(passed + inline_providers)


def _probe_endpoint(
    base_url: str,
    probe_path: str,
    requires_key: bool,
    key_env: str | None,
    extra_headers: dict[str, str] | None = None,
    api_key_header: str | None = None,
) -> _ProbeStatus:
    """Perform a lightweight HTTP probe for a provider."""
    if not base_url:
        return _ProbeStatus.UNREACHABLE

    url = urljoin(base_url.rstrip("/") + "/", probe_path)

    headers: dict[str, str] = dict(extra_headers or {})
    if requires_key:
        if not key_env or not os.getenv(key_env):
            return _ProbeStatus.NO_KEY
        key: str = os.getenv(key_env, "")
        if api_key_header:
            headers[api_key_header] = key
        else:
            headers["Authorization"] = f"Bearer {key}"

    try:
        resp = cast(httpx.Response, httpx.get(url, headers=headers, timeout=2.0))
        if resp.status_code in (401, 403):
            return _ProbeStatus.AUTH
        if resp.status_code < 400:
            return _ProbeStatus.OK
        return _ProbeStatus.UNREACHABLE
    except Exception:
        return _ProbeStatus.UNREACHABLE


class StackLetsGo(Subcommand):
    """Auto-detect providers, generate runtime config, and start the stack (deprecated, use 'ogx letsgo' instead)."""

    def __init__(self, subparsers: Any) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "letsgo",
            prog="ogx stack letsgo",
            description="""Auto-detect providers and start the stack.

NOTE: 'ogx stack letsgo' is deprecated. Use 'ogx letsgo' instead.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_lets_go_cmd)

    def _add_arguments(self) -> None:
        add_letsgo_arguments(self.parser)

    def _run_stack_lets_go_cmd(self, args: argparse.Namespace) -> None:
        warnings.warn(
            "'ogx stack letsgo' is deprecated and will be removed in a future release. Use 'ogx letsgo' instead.",
            FutureWarning,
            stacklevel=1,
        )
        run_letsgo_cmd(args, self.parser)
