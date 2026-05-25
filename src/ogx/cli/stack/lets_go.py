# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import asyncio
import enum
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import yaml
from termcolor import cprint

from ogx.cli.stack.run import _start_ui_development_server, _uvicorn_run
from ogx.cli.subcommand import Subcommand
from ogx.core.build import get_provider_dependencies
from ogx.core.distribution import get_provider_registry
from ogx.core.stack import replace_env_vars, run_config_from_dynamic_config_spec
from ogx.core.utils.config_dirs import DISTRIBS_BASE_DIR
from ogx.core.utils.dynamic import instantiate_class_type
from ogx.log import get_logger
from ogx_api import Api, RemoteProviderSpec
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
    MISSING_DEPS = "missing_deps"
    UNREACHABLE = "unreachable"
    NEEDS_KEY = "needs_key"  # reachable but optional auth token not configured


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

    Each provider is probed by instantiating it and calling list_models() to confirm
    availability and model access. Providers that require an API key skip probing
    when the key environment variable is not set.
    """
    candidates = [
        # provider_type, base_url_env, default_base_url, required_api_key_env, optional_api_key_env
        ("remote::ollama", "OLLAMA_URL", "http://localhost:11434/v1", None, None),
        ("remote::vllm", "VLLM_URL", "http://localhost:8000/v1", None, "VLLM_API_TOKEN"),
        ("remote::llama-cpp-server", "LLAMA_CPP_SERVER_URL", "http://localhost:8080/v1", None, None),
        ("remote::openai", "OPENAI_BASE_URL", "https://api.openai.com/v1", "OPENAI_API_KEY", None),
        (
            "remote::llama-openai-compat",
            "LLAMA_API_BASE_URL",
            "https://api.llama.com/compat/v1/",
            "LLAMA_API_KEY",
            None,
        ),
        ("remote::anthropic", None, "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", None),
        ("remote::gemini", None, "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", None),
        ("remote::azure", "AZURE_API_BASE", "", "AZURE_API_KEY", None),
    ]

    passed: list[str] = []
    missing_deps_providers: dict[str, list[str]] = {}  # provider_type -> pip_packages
    cprint("Scanning for available providers...", color="cyan")
    for provider_type, base_url_env, default_base_url, required_api_key_env, optional_api_key_env in candidates:
        status, model_count, base_url, base_source, pip_packages = _probe_provider_availability(
            provider_type, base_url_env, default_base_url, required_api_key_env, optional_api_key_env
        )

        # Build annotation parts
        parts = []
        if base_url:
            parts.append(f"{base_url}, {base_source}")
        if required_api_key_env:
            parts.append(f"{required_api_key_env} {'set' if os.getenv(required_api_key_env) else 'not set'}")
        if optional_api_key_env:
            parts.append(f"{optional_api_key_env} {'set' if os.getenv(optional_api_key_env) else 'not set'}")
        annotation = ", ".join(parts) if parts else ""

        if status == _ProbeStatus.MISSING_DEPS and pip_packages:
            missing_deps_providers[provider_type] = pip_packages

        if status == _ProbeStatus.OK:
            passed.append(f"inference={provider_type}")
            if annotation:
                cprint(f"  ✓ {provider_type} ({annotation}) — {model_count} models", color="green")
            else:
                cprint(f"  ✓ {provider_type} ({model_count} models)", color="green")
        elif status == _ProbeStatus.NO_KEY:
            if annotation:
                cprint(f"  ✗ {provider_type} ({annotation})", color="yellow")
            else:
                cprint(f"  ✗ {provider_type}", color="yellow")
        elif status == _ProbeStatus.NEEDS_KEY:
            passed.append(f"inference={provider_type}")
            if annotation:
                cprint(f"  ⊘ {provider_type} ({annotation}) — optional token not configured", color="cyan")
            else:
                cprint(f"  ⊘ {provider_type} — optional token not configured", color="cyan")
        elif status == _ProbeStatus.AUTH:
            if annotation:
                cprint(f"  ✗ {provider_type} ({annotation}) — auth error", color="yellow")
            else:
                cprint(f"  ✗ {provider_type} — auth error", color="yellow")
        elif status == _ProbeStatus.MISSING_DEPS:
            if annotation:
                cprint(f"  ✗ {provider_type} ({annotation}) — missing dependencies", color="yellow")
            else:
                cprint(f"  ✗ {provider_type} — missing dependencies", color="yellow")
        else:
            if annotation:
                cprint(f"  ✗ {provider_type} ({annotation}) — unreachable", color="yellow")
            else:
                cprint(f"  ✗ {provider_type} — unreachable", color="yellow")

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
        if missing_deps_providers:
            cprint("Available providers with missing dependencies:", color="cyan")
            for provider_type, pip_packages in missing_deps_providers.items():
                packages_str = " ".join(f"'{pkg}'" for pkg in pip_packages)
                cprint(f"  {provider_type}: uv pip install {packages_str}", color="cyan")
    else:
        cprint("\nDetected no inference providers, not starting stack.", color="red")
        if missing_deps_providers:
            cprint("Install missing dependencies:", color="cyan")
            for provider_type, pip_packages in missing_deps_providers.items():
                packages_str = " ".join(f"'{pkg}'" for pkg in pip_packages)
                cprint(f"  {provider_type}: uv pip install {packages_str}", color="cyan")
    return ",".join(passed + inline_providers)


async def _list_models_with_timeout(provider: Any, timeout_seconds: float = 5) -> list[Any] | None:
    """Call list_models with timeout and proper error handling."""
    if not hasattr(provider, "list_models"):
        return None
    try:
        models = await asyncio.wait_for(provider.list_models(), timeout=timeout_seconds)
        return list(models) if models else []
    except TimeoutError:
        raise


async def _instantiate_with_timeout(factory_fn: Any, config: Any) -> Any:
    """Call factory function with timeout."""
    try:
        result = factory_fn(config, {})
        if asyncio.iscoroutine(result):
            return await asyncio.wait_for(result, timeout=5.0)
        return result
    except TimeoutError:
        raise


def _is_auth_error(e: Exception) -> bool:
    """Return True if the exception represents an authentication/authorization failure."""
    error_str = str(e).lower()
    return (
        "401" in error_str
        or "403" in error_str
        or "unauthorized" in error_str
        or "forbidden" in error_str
        or "invalid_api_key" in error_str
    )


def _probe_provider_availability(
    provider_type: str,
    base_url_env: str | None,
    default_base_url: str,
    required_api_key_env: str | None,
    optional_api_key_env: str | None = None,
) -> tuple[_ProbeStatus, int, str, str, list[str] | None]:
    """Instantiate a provider and probe availability by listing models.

    Args:
        provider_type: Provider type string (e.g., "remote::openai")
        base_url_env: Environment variable name for base URL override, or None
        default_base_url: Default base URL if env var not set
        required_api_key_env: Environment variable name for required API key, or None
        optional_api_key_env: Environment variable name for optional API key, or None

    Returns:
        Tuple of (status, model_count, base_url, base_source, pip_packages).
        base_source is "default" or the env var name. base_url is empty string if not applicable.
        pip_packages is a list of packages to install for MISSING_DEPS, None otherwise.
        Returns NEEDS_KEY if required key is present but optional key is missing.
    """
    # Determine base URL and source
    base_url = ""
    base_source = ""
    if base_url_env:
        env_val = os.getenv(base_url_env)
        if env_val:
            base_url = env_val
            base_source = f"from {base_url_env}"
        else:
            base_url = default_base_url
            base_source = "default"
    elif default_base_url:
        base_url = default_base_url
        base_source = "default"

    # Check if required API key is available
    if required_api_key_env and not os.getenv(required_api_key_env):
        return _ProbeStatus.NO_KEY, 0, base_url, base_source, None

    # Track if optional API key is missing (provider works but with reduced functionality)
    optional_key_missing = optional_api_key_env and not os.getenv(optional_api_key_env)

    try:
        # Load provider registry and look up the spec
        registry = get_provider_registry()
        if Api.inference not in registry or provider_type not in registry[Api.inference]:
            logger.debug("Provider not found in registry", provider_type=provider_type)
            return _ProbeStatus.UNREACHABLE, 0, base_url, base_source, None

        provider_spec = registry[Api.inference][provider_type]
        logger.debug("Found provider in registry", provider_type=provider_type, module=provider_spec.module)

        # Get config defaults
        try:
            config_class = instantiate_class_type(provider_spec.config_class)
            config_defaults = config_class.sample_run_config(__distro_dir__=tempfile.gettempdir())
            logger.debug("Got config defaults", provider_type=provider_type)

            # Substitute environment variables in config (e.g., ${env.VAR_NAME:=default})
            config_defaults = replace_env_vars(config_defaults)
        except ModuleNotFoundError as e:
            logger.debug("Provider dependencies not installed", provider_type=provider_type, module=str(e)[:200])
            return _ProbeStatus.MISSING_DEPS, 0, base_url, base_source, provider_spec.pip_packages
        except Exception as e:
            logger.debug("Failed to get config defaults", provider_type=provider_type, error=str(e)[:200])
            return _ProbeStatus.UNREACHABLE, 0, base_url, base_source, None

        # Instantiate config object
        try:
            config = config_class(**config_defaults)
            logger.debug("Instantiated config", provider_type=provider_type)
        except ModuleNotFoundError as e:
            logger.debug("Provider dependencies not installed", provider_type=provider_type, module=str(e)[:200])
            return _ProbeStatus.MISSING_DEPS, 0, base_url, base_source, provider_spec.pip_packages
        except Exception as e:
            logger.debug("Failed to instantiate config", provider_type=provider_type, error=str(e)[:200])
            return _ProbeStatus.UNREACHABLE, 0, base_url, base_source, None

        # Import provider module and instantiate
        if not provider_spec.module:
            logger.debug("Provider spec missing module", provider_type=provider_type)
            return _ProbeStatus.UNREACHABLE, 0, base_url, base_source, None

        try:
            module = importlib.import_module(provider_spec.module)
            logger.debug("Imported provider module", provider_type=provider_type, module=provider_spec.module)
        except ModuleNotFoundError as e:
            logger.debug("Provider dependencies not installed", provider_type=provider_type, module=str(e)[:200])
            return _ProbeStatus.MISSING_DEPS, 0, base_url, base_source, provider_spec.pip_packages
        except Exception as e:
            logger.debug("Failed to import provider module", module=provider_spec.module, error=str(e)[:200])
            return _ProbeStatus.UNREACHABLE, 0, base_url, base_source, None

        try:
            # Call appropriate factory function
            if isinstance(provider_spec, RemoteProviderSpec):
                method_name = "get_adapter_impl"
            else:  # InlineProviderSpec
                method_name = "get_provider_impl"

            factory_fn = getattr(module, method_name)
            logger.debug("Calling factory function", provider_type=provider_type, method=method_name)
            # Pass empty deps dict {} for single-provider probing
            provider = asyncio.run(_instantiate_with_timeout(factory_fn, config))
            logger.debug("Provider instantiated successfully", provider_type=provider_type)

            # Set required attributes (normally done by resolver)
            provider.__provider_id__ = provider_type
            provider.__provider_spec__ = provider_spec
            provider.__provider_config__ = config
        except ModuleNotFoundError as e:
            logger.debug("Provider dependencies not installed", provider_type=provider_type, module=str(e)[:200])
            return _ProbeStatus.MISSING_DEPS, 0, base_url, base_source, provider_spec.pip_packages
        except Exception as e:
            logger.debug("Failed to instantiate provider", provider_type=provider_type, error=str(e)[:300])
            if _is_auth_error(e):
                logger.warning(
                    "Provider auth failed", provider_type=provider_type, base_url=base_url, error=str(e)[:200]
                )
                return _ProbeStatus.AUTH, 0, base_url, base_source, None
            return _ProbeStatus.UNREACHABLE, 0, base_url, base_source, None

        # List models with timeout
        try:
            logger.debug("Calling list_models", provider_type=provider_type)
            models = asyncio.run(_list_models_with_timeout(provider, timeout_seconds=5))
            model_count = len(models) if models else 0
            logger.debug("Listed models successfully", provider_type=provider_type, model_count=model_count)

            # Cleanup provider
            if hasattr(provider, "aclose"):
                try:
                    asyncio.run(provider.aclose())
                except Exception as e:
                    logger.debug("Failed to cleanup provider", provider_type=provider_type, error=str(e)[:200])

            if model_count == 0:
                logger.debug("Provider returned zero models", provider_type=provider_type)
                return _ProbeStatus.UNREACHABLE, 0, base_url, base_source, None
            # Return NEEDS_KEY if optional API key is missing, otherwise OK
            status = _ProbeStatus.NEEDS_KEY if optional_key_missing else _ProbeStatus.OK
            return status, model_count, base_url, base_source, None
        except TimeoutError:
            logger.debug("Model listing timed out", provider_type=provider_type)
            return _ProbeStatus.UNREACHABLE, 0, base_url, base_source, None
        except Exception as e:
            logger.debug("Failed to list models", provider_type=provider_type, error=str(e)[:300])
            if _is_auth_error(e):
                logger.warning(
                    "Provider auth failed during model listing",
                    provider_type=provider_type,
                    base_url=base_url,
                    error=str(e)[:200],
                )
                return _ProbeStatus.AUTH, 0, base_url, base_source, None
            return _ProbeStatus.UNREACHABLE, 0, base_url, base_source, None
    except Exception as e:
        logger.debug("Unexpected error during provider probing", provider_type=provider_type, error=str(e)[:300])
        return _ProbeStatus.UNREACHABLE, 0, base_url, base_source, None


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
