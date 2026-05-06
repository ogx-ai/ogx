# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import concurrent.futures
import os
import sys
import traceback
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version as parse_version
from pathlib import Path
from typing import Any

import httpx
import yaml
import zstandard
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import BadRequestError
from starlette.types import ASGIApp, Receive, Scope, Send

from ogx.core.access_control.access_control import AccessDeniedError
from ogx.core.datatypes import (
    AuthenticationRequiredError,
    StackConfig,
    process_cors_config,
)
from ogx.core.distribution import builtin_automatically_routed_apis
from ogx.core.exceptions import translate_exception
from ogx.core.external import load_external_apis
from ogx.core.request_headers import (
    request_provider_data_context,
    user_from_scope,
)
from ogx.core.server.fastapi_router_registry import (
    _ROUTER_FACTORIES,
    build_fastapi_router,
    register_external_api_routers,
)
from ogx.core.stack import (
    Stack,
    cast_distro_name_to_string,
    replace_env_vars,
)
from ogx.core.utils.config import redact_sensitive_fields
from ogx.core.utils.config_resolution import resolve_config_or_distro
from ogx.log import LoggingConfig, get_logger, parse_yaml_config, setup_logging
from ogx_api import Api, ConflictError, ResourceNotFoundError
from ogx_api.common.errors import OpenAIErrorResponse

from .auth import AuthenticationMiddleware, RouteAuthorizationMiddleware
from .metrics import RequestMetricsMiddleware, build_route_to_api_map
from .quota import QuotaMiddleware

REPO_ROOT = Path(__file__).parent.parent.parent.parent

logger = get_logger(name=__name__, category="core::server")


def warn_with_traceback(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any = None,
    line: str | None = None,
) -> None:
    """Custom warning handler that prints a full stack traceback alongside the warning."""
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


if os.environ.get("OGX_TRACE_WARNINGS"):
    warnings.showwarning = warn_with_traceback


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions by translating them to JSON error responses.

    Args:
        request: The incoming HTTP request.
        exc: The unhandled exception.

    Returns:
        A JSONResponse with the appropriate HTTP status code and error message.
    """
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    http_exc = translate_exception(exc)

    # OpenAI-compat Vector Stores endpoints treat many "not found" conditions as 400s.
    # Our core exceptions model these as ResourceNotFoundError (mapped to 404 by default),
    # but integration tests (and OpenAI client behavior expectations in this repo)
    # assert they surface as BadRequestError instead.
    if isinstance(exc, ResourceNotFoundError) and request.url.path.startswith("/v1/vector_stores"):
        http_exc = HTTPException(status_code=httpx.codes.BAD_REQUEST, detail=str(exc))

    return JSONResponse(
        status_code=http_exc.status_code, content=OpenAIErrorResponse.from_message(http_exc.detail).to_dict()
    )


class StackApp(FastAPI):
    """
    A wrapper around the FastAPI application to hold a reference to the Stack instance so that we can
    start background tasks (e.g. refresh model registry periodically) from the lifespan context manager.
    """

    def __init__(self, config: StackConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stack: Stack = Stack(config)

        # Initialize stack in a temporary event loop to set up impls for route registration.
        # Storage backends use lazy engine initialization, so connections are created on
        # first use in the correct event loop, avoiding event loop mismatch issues.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, self.stack.initialize())  # type: ignore[no-untyped-call]
            future.result()


@asynccontextmanager
async def lifespan(app: StackApp) -> AsyncIterator[None]:
    """FastAPI lifespan context manager that starts background tasks and handles shutdown.

    Args:
        app: The StackApp instance.
    """
    server_version = parse_version("ogx")

    logger.info("Starting up OGX server", version=server_version)
    assert app.stack is not None
    app.stack.create_registry_refresh_task()  # type: ignore[no-untyped-call]
    yield
    logger.info("Shutting down")
    await app.stack.shutdown()  # type: ignore[no-untyped-call]


async def _send_error_response(send: Send, status: int, message: str) -> None:
    """Send an ASGI error response with an OpenAI-compatible error body."""
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [[b"content-type", b"application/json"]],
        }
    )
    error_msg = OpenAIErrorResponse.from_message(message).to_bytes()
    await send({"type": "http.response.body", "body": error_msg})


class ClientVersionMiddleware:
    """ASGI middleware that rejects requests from clients with incompatible major.minor versions."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.server_version = parse_version("ogx")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> Any:
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            client_version = headers.get(b"x-ogx-client-version", b"").decode()
            if client_version:
                try:
                    client_version_parts = tuple(map(int, client_version.split(".")[:2]))
                    server_version_parts = tuple(map(int, self.server_version.split(".")[:2]))
                    if client_version_parts != server_version_parts:
                        return await _send_error_response(
                            send,
                            status=httpx.codes.UPGRADE_REQUIRED,
                            message=f"Client version {client_version} is not compatible with server version {self.server_version}. Please update your client.",
                        )
                except (ValueError, IndexError):
                    # If version parsing fails, let the request through
                    pass

        return await self.app(scope, receive, send)


class ProviderDataMiddleware:
    """Middleware to set up request context for all routes.

    Sets up provider data context from X-OGX-Provider-Data header
    and auth attributes. Also handles test context propagation when
    running in test mode for deterministic ID generation.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> Any:
        if scope["type"] == "http":
            headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}
            user = user_from_scope(dict(scope))

            with request_provider_data_context(headers, user):
                test_context_token = None
                reset_fn = None
                if os.environ.get("OGX_TEST_INFERENCE_MODE"):
                    from ogx.core.testing_context import (
                        reset_test_context,
                        sync_test_context_from_provider_data,
                    )

                    test_context_token = sync_test_context_from_provider_data()  # type: ignore[no-untyped-call]
                    reset_fn = reset_test_context
                try:
                    return await self.app(scope, receive, send)
                finally:
                    if test_context_token and reset_fn:
                        reset_fn(test_context_token)

        return await self.app(scope, receive, send)


def create_app() -> StackApp:
    """Create and configure the FastAPI application.

    This factory function reads configuration from environment variables:
    - OGX_CONFIG: Path to config file (required)

    Returns:
        Configured StackApp instance.
    """
    config_file_str = os.getenv("OGX_CONFIG")
    if config_file_str is None:
        raise ValueError("OGX_CONFIG environment variable is required")

    config_file = resolve_config_or_distro(config_file_str)

    # Load and process configuration
    logger_config = None
    with open(config_file) as fp:
        config_contents = yaml.safe_load(fp)
        if isinstance(config_contents, dict) and (cfg := config_contents.get("logging_config")):
            logger_config = LoggingConfig(**cfg)

        # Configure logging in each worker process
        if logger_config:
            category_levels = parse_yaml_config(logger_config)
            setup_logging(category_levels)
        else:
            setup_logging()

        logger = get_logger(name=__name__, category="core::server", config=logger_config)

        config = replace_env_vars(config_contents)
        config = StackConfig(**cast_distro_name_to_string(config))

    _log_run_config(run_config=config)

    app = StackApp(
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        config=config,
    )

    if not os.environ.get("OGX_DISABLE_VERSION_CHECK"):
        app.add_middleware(ClientVersionMiddleware)

    app.add_middleware(ProviderDataMiddleware)

    impls = app.stack.impls
    assert impls is not None

    if config.server.auth:
        # Add route authorization middleware if route_policy is configured
        # This can work independently of authentication
        # NOTE: Add this FIRST because middleware wraps in reverse order (last added runs first)
        # We want: Request → Auth → RouteAuth → App
        if config.server.auth.route_policy:
            logger.info("Enabling route-level authorization", rule_count=len(config.server.auth.route_policy))
            app.add_middleware(RouteAuthorizationMiddleware, route_policy=config.server.auth.route_policy)

        # Add authentication middleware only if provider is configured
        # This runs FIRST in the middleware chain (last added = first to run)
        if config.server.auth.provider_config:
            logger.info("Enabling authentication", provider=config.server.auth.provider_config.type.value)
            app.add_middleware(AuthenticationMiddleware, auth_config=config.server.auth, impls=impls)
    else:
        if config.server.quota:
            quota = config.server.quota
            logger.warning(
                "Configured authenticated_max_requests (%d) but no auth is enabled; "
                "falling back to anonymous_max_requests (%d) for all the requests",
                quota.authenticated_max_requests,
                quota.anonymous_max_requests,
            )

    if config.server.quota:
        logger.info("Enabling quota middleware for authenticated and anonymous clients")

        quota = config.server.quota
        anonymous_max_requests = quota.anonymous_max_requests
        # if auth is disabled, use the anonymous max requests
        authenticated_max_requests = quota.authenticated_max_requests if config.server.auth else anonymous_max_requests

        kv_config = quota.kvstore
        window_map = {"day": 86400}
        window_seconds = window_map[quota.period.value]

        app.add_middleware(
            QuotaMiddleware,
            kv_config=kv_config,
            anonymous_max_requests=anonymous_max_requests,
            authenticated_max_requests=authenticated_max_requests,
            window_seconds=window_seconds,
        )

    if config.server.cors:
        logger.info("Enabling CORS")
        cors_config = process_cors_config(config.server.cors)
        if cors_config:
            app.add_middleware(CORSMiddleware, **cors_config.model_dump())

    # Load and register external API routers if configured
    external_apis = load_external_apis(config)
    if external_apis:
        register_external_api_routers(external_apis)

    if config.apis:
        apis_to_serve = set(config.apis)
    else:
        apis_to_serve = set(impls.keys())

    for inf in builtin_automatically_routed_apis():
        # if we do not serve the corresponding router API, we should not serve the routing table API
        if inf.router_api.value not in apis_to_serve:
            continue
        apis_to_serve.add(inf.routing_table_api.value)

    apis_to_serve.add("admin")
    apis_to_serve.add("inspect")
    apis_to_serve.add("providers")
    apis_to_serve.add("prompts")
    apis_to_serve.add("conversations")
    apis_to_serve.add("connectors")

    # Build route-to-API mapping and add request metrics middleware.
    # Added last so it runs first (outermost), wrapping auth/quota/cors.
    route_to_api = build_route_to_api_map(_ROUTER_FACTORIES, impls)
    app.add_middleware(RequestMetricsMiddleware, route_to_api=route_to_api)

    for api_str in apis_to_serve:
        api = Api(api_str)
        impl = impls[api]
        router = build_fastapi_router(api, impl)
        if router:
            app.include_router(router)
            logger.debug("Registered FastAPI router", api=str(api))

    logger.debug("Serving APIs", apis=list(apis_to_serve))

    # Decompress zstd-encoded request bodies (e.g. from Codex CLI)
    # Must be a raw ASGI middleware to intercept the body before Starlette reads it
    class ZstdDecompressionMiddleware:
        def __init__(self, app: ASGIApp) -> None:
            self.app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> Any:
            if scope["type"] != "http":
                return await self.app(scope, receive, send)

            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            content_encoding = headers.get(b"content-encoding", b"").decode().lower()

            if content_encoding != "zstd":
                return await self.app(scope, receive, send)

            # Collect the full request body first (needed for both success and fallback)
            body_parts: list[bytes] = []
            while True:
                message = await receive()
                body_parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break

            compressed_body = b"".join(body_parts)

            try:
                max_decompressed_size = 100 * 1024 * 1024  # 100 MB
                decompressor = zstandard.ZstdDecompressor()
                # Use streaming decompression to handle frames without content size
                reader = decompressor.stream_reader(compressed_body)
                decompressed_body = reader.read(max_decompressed_size)
                if reader.read(1):
                    reader.close()
                    return await _send_error_response(
                        send,
                        status=413,
                        message=f"Decompressed request body exceeds maximum allowed size of {max_decompressed_size} bytes",
                    )
                reader.close()

                # Strip content-encoding header and update content-length
                new_headers = [
                    (k, v) for k, v in scope["headers"] if k.lower() not in (b"content-encoding", b"content-length")
                ]
                new_headers.append((b"content-length", str(len(decompressed_body)).encode()))
                scope["headers"] = new_headers

                # Feed the decompressed body back, then delegate to the
                # original receive for disconnect detection so streaming
                # responses stay alive until the client actually disconnects.
                body_sent = False

                async def receive_decompressed() -> dict:  # type: ignore[type-arg]
                    nonlocal body_sent
                    if not body_sent:
                        body_sent = True
                        return {"type": "http.request", "body": decompressed_body, "more_body": False}
                    return await receive()

                return await self.app(scope, receive_decompressed, send)
            except Exception as e:
                logger.warning("Failed to decompress zstd request body, falling back to compressed data", error=str(e))

                # Replay the original compressed body since decompression failed
                body_sent = False

                async def receive_original() -> dict:  # type: ignore[type-arg]
                    nonlocal body_sent
                    if not body_sent:
                        body_sent = True
                        return {"type": "http.request", "body": compressed_body, "more_body": False}
                    return await receive()

                return await self.app(scope, receive_original, send)

    app.add_middleware(ZstdDecompressionMiddleware)

    # Register specific exception handlers before the generic Exception handler
    # This prevents the re-raising behavior that causes connection resets
    app.exception_handler(RequestValidationError)(global_exception_handler)
    app.exception_handler(ConflictError)(global_exception_handler)
    app.exception_handler(ResourceNotFoundError)(global_exception_handler)
    app.exception_handler(AuthenticationRequiredError)(global_exception_handler)
    app.exception_handler(AccessDeniedError)(global_exception_handler)
    app.exception_handler(BadRequestError)(global_exception_handler)
    # Generic Exception handler should be last
    app.exception_handler(Exception)(global_exception_handler)

    return app


def _log_run_config(run_config: StackConfig) -> None:
    """Logs the run config with redacted fields and disabled providers removed."""
    logger.info("Stack Configuration:")
    safe_config = redact_sensitive_fields(run_config.model_dump(mode="json"))
    clean_config = remove_disabled_providers(safe_config)
    logger.info(yaml.dump(clean_config, indent=2))


def remove_disabled_providers(obj: Any) -> Any:
    """Recursively remove disabled providers from a configuration dictionary.

    Args:
        obj: A configuration value (dict, list, or scalar).

    Returns:
        The configuration with disabled provider entries removed.
    """
    if isinstance(obj, dict):
        # Filter out items where provider_id is explicitly disabled or empty
        if "provider_id" in obj and obj["provider_id"] in ("__disabled__", "", None):
            return None
        return {k: v for k, v in ((k, remove_disabled_providers(v)) for k, v in obj.items()) if v is not None}
    elif isinstance(obj, list):
        return [item for item in (remove_disabled_providers(i) for i in obj) if item is not None]
    else:
        return obj
