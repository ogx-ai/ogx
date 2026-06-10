# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import asyncio
import contextlib
import enum
import importlib
import inspect
import logging  # allow-direct-logging :: for direct logging control in _suppress_provider_logs
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Awaitable, Callable, Coroutine, Generator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml
from termcolor import cprint

from ogx.cli.stack.run import _start_ui_development_server, _uvicorn_run
from ogx.cli.subcommand import Subcommand
from ogx.core.build import get_provider_dependencies
from ogx.core.datatypes import Provider, QualifiedModel, StackConfig, VectorStoresConfig
from ogx.core.distribution import get_provider_registry
from ogx.core.stack import extract_env_var_references, replace_env_vars, run_config_from_dynamic_config_spec
from ogx.core.utils.config_dirs import DISTRIBS_BASE_DIR
from ogx.core.utils.dynamic import instantiate_class_type
from ogx.log import get_logger
from ogx_api import Api, ModelType
from ogx_api.models.models import ModelInput

logger = get_logger(name=__name__, category="cli")


# Type protocols for provider and config objects
@runtime_checkable
class ProbeableProvider(Protocol):
    """Protocol for providers that support model listing and lifecycle management."""

    async def list_models(self) -> list[Any]:
        """Return list of available models."""
        ...

    def initialize(self) -> Coroutine[Any, Any, None] | None:
        """Initialize provider (optional async)."""
        ...

    def shutdown(self) -> Coroutine[Any, Any, None] | None:
        """Shutdown provider (optional async)."""
        ...


class _FactoryDispatcher:
    """Determines factory function name and retrieves it from a module based on provider spec type.

    Centralizes the isinstance-based dispatch logic, replacing runtime type checks
    with a single point of control.
    """

    @staticmethod
    def method_name_for_spec(spec: Any) -> str:
        """Return the factory method name for a given provider spec.

        Uses isinstance to determine spec type (isolated to this method).
        Returns one of: "get_provider_impl", "get_adapter_impl", "get_auto_router_impl", "get_routing_table_impl".
        """
        # Import here to avoid circular imports at module level
        from ogx.core.datatypes import AutoRoutedProviderSpec, RoutingTableProviderSpec
        from ogx_api import RemoteProviderSpec

        if isinstance(spec, RemoteProviderSpec):
            return "get_adapter_impl"
        elif isinstance(spec, AutoRoutedProviderSpec):
            return "get_auto_router_impl"
        elif isinstance(spec, RoutingTableProviderSpec):
            return "get_routing_table_impl"
        else:  # Default: InlineProviderSpec
            return "get_provider_impl"

    @staticmethod
    def get_factory(spec: Any, module: Any) -> Callable[[Any, dict[str, Any]], Awaitable[Any]] | None:
        """Retrieve factory function from module for the given spec.

        Args:
            spec: Provider spec (ProviderSpec subclass).
            module: Imported module containing factory functions.

        Returns:
            Factory function callable or None if not found.
        """
        method_name = _FactoryDispatcher.method_name_for_spec(spec)
        return getattr(module, method_name, None)


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
    parser.add_argument(
        "--default-embedding-model",
        type=str,
        default=None,
        metavar="PROVIDER_ID/MODEL_ID",
        help="Default embedding model for vector stores, in the form 'provider_id/model_id' (e.g. 'sentence-transformers/nomic-ai/nomic-embed-text-v1.5'). When omitted, the server auto-detects an embedding model from registered providers.",
    )
    parser.add_argument(
        "--default-embedding-dimension",
        type=int,
        default=None,
        metavar="DIMENSION",
        help="Embedding dimension for the default embedding model. Required when --default-embedding-model is specified.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging during provider scanning and server startup.",
    )


def _add_file_search_and_responses(run_config: StackConfig) -> None:
    """Add file-search and responses providers to the stack config.

    These are only added after confirming an embedding model is available.
    """
    # Add tool_runtime API with file-search provider
    if "tool_runtime" not in run_config.providers:
        run_config.providers["tool_runtime"] = []
    if not any(p.provider_type == "inline::file-search" for p in run_config.providers["tool_runtime"]):
        run_config.providers["tool_runtime"].append(
            Provider(provider_id="file-search", provider_type="inline::file-search")
        )
    # Add tool_runtime to APIs if not already present
    if "tool_runtime" not in run_config.apis:
        run_config.apis.append("tool_runtime")

    # Add responses API with builtin provider
    if "responses" not in run_config.providers:
        run_config.providers["responses"] = []
    if not any(p.provider_type == "inline::builtin" for p in run_config.providers["responses"]):
        run_config.providers["responses"].append(
            Provider(
                provider_id="builtin",
                provider_type="inline::builtin",
                config={
                    "persistence": {
                        "responses": {
                            "table_name": "responses",
                            "backend": "sql_default",
                        }
                    }
                },
            )
        )
    # Add responses to APIs if not already present
    if "responses" not in run_config.apis:
        run_config.apis.append("responses")

    # Add web search providers in priority order: brave -> tavily -> bing
    _web_search_order = [
        ("remote::brave-search", "brave-search"),
        ("remote::tavily-search", "tavily-search"),
        ("remote::bing-search", "bing-search"),
    ]
    tool_runtime_registry = get_provider_registry().get(Api.tool_runtime, {})
    existing_web_search: set[str] = {
        p.provider_type
        for p in run_config.providers["tool_runtime"]
        if p.provider_type in {pt for pt, _ in _web_search_order}
    }

    added = False
    all_env_vars: list[str] = []

    for provider_type, provider_id in _web_search_order:
        if provider_type in existing_web_search:
            cprint(f"  ✓ {provider_id} (web search)", color="green")
            added = True
            continue

        spec = tool_runtime_registry.get(provider_type)
        if spec is None:
            continue

        try:
            config_class = instantiate_class_type(spec.config_class)
            config_template = config_class.sample_run_config(__distro_dir__=tempfile.gettempdir())
        except Exception:
            cprint(f"  ✗ {provider_id} (web search) — failed to construct config template", color="yellow")
            continue

        env_vars_in_template = extract_env_var_references(config_template)
        all_env_vars.extend(env_vars_in_template)

        if not any(os.environ.get(v) for v in env_vars_in_template):
            continue

        try:
            resolved_config = replace_env_vars(config_template)
            config_class(**resolved_config)  # validate construction
        except Exception:
            cprint(f"  ✗ {provider_id} (web search) — failed to construct config with env vars", color="yellow")
            continue  # noqa S112 -- exception is reported to the user via cprint before continuing

        run_config.providers["tool_runtime"].append(
            Provider(
                provider_id=provider_id,
                provider_type=provider_type,
                config=resolved_config,
            )
        )
        cprint(f"  ✓ {provider_id} (web search)", color="green")
        added = True

    if not added:
        if all_env_vars:
            vars_str = ", ".join(dict.fromkeys(all_env_vars))
            cprint(
                f"  ✗ web search disabled (no API key set — set {vars_str})",
                color="yellow",
            )
        else:
            cprint("  ✗ web search disabled", color="yellow")

    cprint("  ✓ inline::file-search (built-in)", color="green")
    cprint("  ✓ inline::builtin responses (built-in)", color="green")


def run_letsgo_cmd(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.enable_ui:
        try:
            _start_ui_development_server(args.port)
        except Exception:
            logger.warning("Failed to start UI development server", exc_info=True)

    if args.providers_override:
        providers_spec = args.providers_override
        autodetect_embedding: tuple[QualifiedModel, int | None] | None = None
    else:
        providers_spec, autodetect_embedding = _autodetect_providers(debug=getattr(args, "debug", False))

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

    if args.default_embedding_model:
        if not args.default_embedding_dimension:
            cprint(
                "Failed: --default-embedding-dimension is required when --default-embedding-model is specified",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)
        provider_id, _, model_id = args.default_embedding_model.partition("/")
        if not model_id:
            cprint(
                f"Failed to parse --default-embedding-model '{args.default_embedding_model}': expected format 'provider_id/model_id'",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)
        existing = run_config.vector_stores or VectorStoresConfig()
        run_config.vector_stores = existing.model_copy(
            update={"default_embedding_model": QualifiedModel(provider_id=provider_id, model_id=model_id)}
        )
        # Explicitly register the model as embedding type so the startup validator and
        # vector store creation can find it, even when the provider doesn't pre-classify it.
        run_config.registered_resources.models.append(
            ModelInput(
                model_id=model_id,
                provider_id=provider_id,
                provider_model_id=model_id,
                model_type=ModelType.embedding,
                metadata={"embedding_dimension": args.default_embedding_dimension},
            )
        )
        cprint(
            f"  ✓ Default embedding model → {args.default_embedding_model} ({args.default_embedding_dimension}d)",
            color="green",
        )
        # Add file-search and responses now that embedding model is confirmed
        _add_file_search_and_responses(run_config)
    elif "vector_io" in run_config.providers:
        detected_result = (
            autodetect_embedding if autodetect_embedding is not None else _detect_embedding_model(run_config)
        )
        if detected_result:
            detected, embedding_dimension = detected_result
            if embedding_dimension is None:
                cprint(
                    "  ✗ Auto-detected embedding model has no dimension — vector stores, file-search, and responses disabled.",
                    color="yellow",
                )
                cprint(
                    "    To enable them, run with --default-embedding-model PROVIDER_ID/MODEL_ID --default-embedding-dimension DIMENSION.",
                    color="yellow",
                )
                run_config.providers.pop("vector_io", None)
                run_config.apis = [a for a in run_config.apis if a != "vector_io"]
            else:
                existing = run_config.vector_stores or VectorStoresConfig()
                run_config.vector_stores = existing.model_copy(update={"default_embedding_model": detected})
                # Keep auto-detected embeddings consistent with --default-embedding-model:
                # register them explicitly as embedding models so startup validation
                # does not rely on provider-side model typing.
                run_config.registered_resources.models.append(
                    ModelInput(
                        model_id=detected.model_id,
                        provider_id=detected.provider_id,
                        provider_model_id=detected.model_id,
                        model_type=ModelType.embedding,
                        metadata={"embedding_dimension": embedding_dimension},
                    )
                )
                cprint(
                    f"  ✓ Auto-detected embedding model → {detected.provider_id}/{detected.model_id} ({embedding_dimension}d)",
                    color="green",
                )
                # Add file-search and responses now that embedding model is confirmed
                _add_file_search_and_responses(run_config)
        else:
            cprint(
                "  ✗ No embedding model detected — vector stores, file-search, and responses disabled.",
                color="yellow",
            )
            cprint(
                "    To enable them, run with --default-embedding-model PROVIDER_ID/MODEL_ID --default-embedding-dimension DIMENSION.",
                color="yellow",
            )
            run_config.providers.pop("vector_io", None)
            run_config.apis = [a for a in run_config.apis if a != "vector_io"]
    else:
        cprint(
            "  ✗ No vector store provider detected — file-search and responses disabled.",
            color="yellow",
        )
        cprint(
            "    To enable them, ensure a vector store is configured and add --default-embedding-model PROVIDER_ID/MODEL_ID.",
            color="yellow",
        )

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


def _autodetect_providers(debug: bool = False) -> tuple[str, tuple[QualifiedModel, int | None] | None]:
    """Probe all candidate providers and return a spec string and first detected embedding model.

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
    detected_embedding: tuple[QualifiedModel, int | None] | None = None
    cprint("Scanning for available providers...", color="cyan")
    for provider_type, base_url_env, default_base_url, required_api_key_env, optional_api_key_env in candidates:
        status, models, base_url, base_source, pip_packages = _probe_provider_availability(
            provider_type, base_url_env, default_base_url, required_api_key_env, optional_api_key_env, debug=debug
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
                cprint(f"  ✓ {provider_type} ({annotation}) — {len(models)} models", color="green")
            else:
                cprint(f"  ✓ {provider_type} ({len(models)} models)", color="green")
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

        if status in (_ProbeStatus.OK, _ProbeStatus.NEEDS_KEY) and detected_embedding is None:
            detected_embedding = _pick_embedding_from_models(models, provider_type.split("::")[-1])

    # Inline providers require no external service — always include them.
    # Note: file-search and responses require an embedding model, so they are added later
    # after confirming an embedding model is available (either user-provided or auto-detected)
    inline_providers = [
        "files=inline::localfs",
        "vector_io=inline::faiss",
        "batches=inline::reference",
        "file_processors=inline::auto",
        "messages=inline::builtin",
    ]
    cprint("  ✓ inline::localfs (built-in)", color="green")
    cprint("  ✓ inline::faiss (built-in)", color="green")
    cprint("  ✓ inline::reference batches (built-in)", color="green")
    cprint("  ✓ inline::auto (built-in)", color="green")
    cprint("  ✓ inline::builtin messages (built-in)", color="green")

    if missing_deps_providers:
        cprint("\nAvailable providers with missing dependencies:", color="cyan")
        for provider_type, pip_packages in missing_deps_providers.items():
            packages_str = " ".join(f"'{pkg}'" for pkg in pip_packages)
            cprint(f"  {provider_type}: uv pip install {packages_str}", color="cyan")

    return ",".join(passed + inline_providers), detected_embedding


async def _list_models_with_timeout(provider: ProbeableProvider, timeout_seconds: float = 5) -> list[Any]:
    """Call list_models with timeout and proper error handling."""
    models = await asyncio.wait_for(provider.list_models(), timeout=timeout_seconds)
    return list(models) if models else []


async def _list_models_from_provider(provider: Any) -> list[Any]:
    """Instantiate an inference provider from a runtime `Provider` entry and return its models.

    This uses the provider registry to resolve the provider spec, instantiates the
    provider implementation via its factory function, then calls `list_models` with
    a timeout-protected helper.
    """
    registry = get_provider_registry()
    spec = registry.get(Api.inference, {}).get(provider.provider_type)
    if spec is None or not spec.module:
        return []

    try:
        module = importlib.import_module(spec.module)
    except Exception:
        return []

    config_type = instantiate_class_type(spec.config_class)
    provider_config = provider.config if isinstance(provider.config, dict) else {}
    try:
        config = config_type(**provider_config)
    except Exception:
        return []

    # Use dispatcher to get factory function
    factory_fn = _FactoryDispatcher.get_factory(spec, module)
    if factory_fn is None:
        return []

    try:
        impl: ProbeableProvider = await _instantiate_with_timeout(factory_fn, config)
    except Exception:
        return []

    # Annotate impl for diagnostic messages and cleanup
    impl.__provider_id__ = provider.provider_id  # type: ignore[attr-defined]
    impl.__provider_spec__ = spec  # type: ignore[attr-defined]

    # Initialize provider if needed
    init_result = impl.initialize()
    if init_result is not None:
        await asyncio.wait_for(init_result, timeout=5.0)

    try:
        models = await _list_models_with_timeout(impl, timeout_seconds=5)
    except Exception:
        models = []

    # Cleanup provider if needed
    shutdown_result = impl.shutdown()
    if shutdown_result is not None:
        await asyncio.wait_for(shutdown_result, timeout=2.0)

    return models or []


def _pick_embedding_from_models(models: list[Any], provider_id: str) -> tuple[QualifiedModel, int | None] | None:
    """Return the best embedding model from a list, preferring one with dimension metadata."""
    best: tuple[QualifiedModel, int | None] | None = None
    for model in models:
        if getattr(model, "model_type", None) != ModelType.embedding:
            continue
        identifier = getattr(model, "identifier", None)
        if not identifier:
            continue
        dimension: int | None = None
        if hasattr(model, "metadata") and isinstance(model.metadata, dict):
            dimension = model.metadata.get("embedding_dimension")
        best = (QualifiedModel(provider_id=provider_id, model_id=str(identifier)), dimension)
        if dimension is not None:
            break
    return best


def _detect_embedding_model(run_config: StackConfig) -> tuple[QualifiedModel, int | None] | None:
    """Find an embedding model by instantiating each inference provider and calling list_models().

    Returns tuple of (QualifiedModel, embedding_dimension) or None if not found.
    Dimension may be None if not available in model metadata — caller must handle this.
    """

    async def _detect_async() -> tuple[QualifiedModel, int | None] | None:
        for provider in run_config.providers.get("inference", []):
            if not provider.provider_id:
                continue
            models = await _list_models_from_provider(provider)
            result = _pick_embedding_from_models(models, provider.provider_id)
            if result is not None:
                return result
        return None

    return asyncio.run(_detect_async())


async def _instantiate_with_timeout(
    factory_fn: Callable[[Any, dict[str, Any]], Awaitable[ProbeableProvider] | ProbeableProvider],
    config: Any,
) -> ProbeableProvider:
    """Call factory function with timeout."""
    try:
        result = factory_fn(config, {})
        if inspect.iscoroutine(result):
            return await asyncio.wait_for(result, timeout=5.0)  # type: ignore[arg-type]
        return result  # type: ignore[return-value]
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


@contextlib.contextmanager
def _suppress_provider_logs(suppress: bool = True) -> Generator[None, None, None]:
    """Context manager to suppress all provider-related logs during probing.

    When suppress=True, temporarily disables all logging. When suppress=False,
    logging is emitted normally. This ensures no logger output appears during provider
    scanning unless the user passes --debug.
    """
    if not suppress:
        yield
        return

    # Disable all logging (using level higher than CRITICAL) to suppress all messages
    previous_disable_level = logging.root.manager.disable
    logging.disable(100)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)


def _probe_provider_availability(
    provider_type: str,
    base_url_env: str | None,
    default_base_url: str,
    required_api_key_env: str | None,
    optional_api_key_env: str | None = None,
    debug: bool = False,
) -> tuple[_ProbeStatus, list[Any], str, str, list[str] | None]:
    """Instantiate a provider and probe availability by listing models.

    Args:
        provider_type: Provider type string (e.g., "remote::openai")
        base_url_env: Environment variable name for base URL override, or None
        default_base_url: Default base URL if env var not set
        required_api_key_env: Environment variable name for required API key, or None
        optional_api_key_env: Environment variable name for optional API key, or None

    Returns:
        Tuple of (status, models, base_url, base_source, pip_packages).
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
        return _ProbeStatus.NO_KEY, [], base_url, base_source, None

    # Track if optional API key is missing (provider works but with reduced functionality)
    optional_key_missing = optional_api_key_env and not os.getenv(optional_api_key_env)

    # Suppress non-critical logging during provider probing unless debug mode is enabled
    with _suppress_provider_logs(suppress=not debug):
        try:
            # Load provider registry and look up the spec
            registry = get_provider_registry()
            if Api.inference not in registry or provider_type not in registry[Api.inference]:
                cprint(f"  ✗ {provider_type} not found in provider registry", color="red")
                return _ProbeStatus.UNREACHABLE, [], base_url, base_source, None

            provider_spec = registry[Api.inference][provider_type]
            logger.debug("Probing provider", provider_type=provider_type, module=provider_spec.module)

            # Get config defaults
            try:
                config_class = instantiate_class_type(provider_spec.config_class)
                config_defaults = config_class.sample_run_config(__distro_dir__=tempfile.gettempdir())
                logger.debug("Loaded config defaults for provider", provider_type=provider_type)

                # Substitute environment variables in config (e.g., ${env.VAR_NAME:=default})
                config_defaults = replace_env_vars(config_defaults)

                # Inject the determined base_url into the config for remote providers
                if base_url and "base_url" in config_defaults:
                    config_defaults["base_url"] = base_url
                    logger.debug("Set base_url in config", provider_type=provider_type, base_url=base_url)
            except ModuleNotFoundError as e:
                cprint(f"    Missing dependencies for {provider_type}: {str(e)[:200]}", color="red")
                return _ProbeStatus.MISSING_DEPS, [], base_url, base_source, provider_spec.pip_packages
            except Exception as e:
                cprint(f"    Failed to get config defaults for {provider_type}: {str(e)[:200]}", color="red")
                return _ProbeStatus.UNREACHABLE, [], base_url, base_source, None

            # Instantiate config object
            try:
                config = config_class(**config_defaults)
                logger.debug("Instantiated config for provider", provider_type=provider_type)
            except ModuleNotFoundError as e:
                cprint(f"    Missing dependencies for {provider_type}: {str(e)[:200]}", color="red")
                return _ProbeStatus.MISSING_DEPS, [], base_url, base_source, provider_spec.pip_packages
            except Exception as e:
                cprint(f"    Failed to instantiate config for {provider_type}: {str(e)[:200]}", color="red")
                return _ProbeStatus.UNREACHABLE, [], base_url, base_source, None

            # Import provider module and instantiate
            if not provider_spec.module:
                cprint(f"    Provider spec missing module for {provider_type}", color="red")
                return _ProbeStatus.UNREACHABLE, [], base_url, base_source, None

            try:
                module = importlib.import_module(provider_spec.module)
                logger.debug("Imported provider module", provider_type=provider_type, module=provider_spec.module)
            except ModuleNotFoundError as e:
                cprint(f"    Missing dependencies for {provider_type}: {str(e)[:200]}", color="red")
                return _ProbeStatus.MISSING_DEPS, [], base_url, base_source, provider_spec.pip_packages
            except Exception as e:
                cprint(f"    Failed to import provider module {provider_spec.module}: {str(e)[:200]}", color="red")
                return _ProbeStatus.UNREACHABLE, [], base_url, base_source, None

            try:
                # Use dispatcher to get factory function
                factory_fn = _FactoryDispatcher.get_factory(provider_spec, module)
                if factory_fn is None:
                    cprint(f"    Failed to find factory function in {provider_spec.module}", color="red")
                    return _ProbeStatus.UNREACHABLE, [], base_url, base_source, None

                logger.debug(
                    "Calling factory function for provider",
                    provider_type=provider_type,
                    factory_name=_FactoryDispatcher.method_name_for_spec(provider_spec),
                )
                # Pass empty deps dict {} for single-provider probing
                provider: ProbeableProvider = asyncio.run(_instantiate_with_timeout(factory_fn, config))  # type: ignore[arg-type]
                logger.debug("Provider instantiated successfully for provider", provider_type=provider_type)

                # Set required attributes (normally done by resolver)
                provider.__provider_id__ = provider_type  # type: ignore[attr-defined]
                provider.__provider_spec__ = provider_spec  # type: ignore[attr-defined]
                provider.__provider_config__ = config  # type: ignore[attr-defined]
            except ModuleNotFoundError:
                return _ProbeStatus.MISSING_DEPS, [], base_url, base_source, provider_spec.pip_packages
            except Exception as e:
                if _is_auth_error(e):
                    return _ProbeStatus.AUTH, [], base_url, base_source, None
                cprint(f"    Failed to instantiate provider {provider_type}: {str(e)[:300]}", color="red")
                return _ProbeStatus.UNREACHABLE, [], base_url, base_source, None

            # List models with timeout
            try:
                logger.debug("Calling list_models for provider", provider_type=provider_type)
                models = asyncio.run(_list_models_with_timeout(provider, timeout_seconds=5))
                logger.debug("Listed models for provider", provider_type=provider_type, model_count=len(models))

                # Cleanup provider: call shutdown() if available.
                try:
                    shutdown_result = provider.shutdown()
                    if shutdown_result is not None:
                        asyncio.run(shutdown_result)  # type: ignore[arg-type]
                except AttributeError:
                    # Provider did not declare `shutdown()`; surface as a warning.
                    cprint(
                        f"    Provider {provider_type} has no declared 'shutdown' method; skipping cleanup",
                        color="red",
                    )
                except Exception as e:
                    cprint(f"    Failed to cleanup provider {provider_type}: {str(e)[:200]}", color="red")

                if not models:
                    cprint(f"    Provider returned zero models for {provider_type}", color="yellow")
                    return _ProbeStatus.UNREACHABLE, [], base_url, base_source, None
                # Return NEEDS_KEY if optional API key is missing, otherwise OK
                status = _ProbeStatus.NEEDS_KEY if optional_key_missing else _ProbeStatus.OK
                return status, models, base_url, base_source, None
            except TimeoutError:
                cprint(f"    Model listing timed out for {provider_type}", color="yellow")
                return _ProbeStatus.UNREACHABLE, [], base_url, base_source, None
            except Exception as e:
                if _is_auth_error(e):
                    return _ProbeStatus.AUTH, [], base_url, base_source, None
                return _ProbeStatus.UNREACHABLE, [], base_url, base_source, None
        except Exception as e:
            cprint(f"  ✗ Unexpected error during provider probing {provider_type}: {str(e)[:300]}", color="red")
            return _ProbeStatus.UNREACHABLE, [], base_url, base_source, None


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
