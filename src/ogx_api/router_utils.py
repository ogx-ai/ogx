# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Utilities for creating FastAPI routers with standard error responses.

This module provides standard error response definitions for FastAPI routers.
These responses use OpenAPI $ref references to component responses defined
in the OpenAPI specification.

It also provides :class:`ExceptionTranslatingRoute`, a reusable route class
that catches known exception types (``ValueError``, ``OGXError``)
and translates them to ``HTTPException`` so that FastAPI returns proper
JSON error responses instead of dropping the connection.
"""

import inspect
from collections.abc import Callable, Coroutine
from typing import Annotated, Any, TypeVar

from fastapi import HTTPException, Path, Query, Request, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel

# OpenAPI extension key to mark routes that don't require authentication.
# Use this in FastAPI route decorators: @router.get("/health", openapi_extra={PUBLIC_ROUTE_KEY: True})
PUBLIC_ROUTE_KEY = "x-public"


standard_responses: dict[int | str, dict[str, Any]] = {
    400: {"$ref": "#/components/responses/BadRequest400"},
    429: {"$ref": "#/components/responses/TooManyRequests429"},
    500: {"$ref": "#/components/responses/InternalServerError500"},
    "default": {"$ref": "#/components/responses/DefaultError"},
}

T = TypeVar("T", bound=BaseModel)


def create_query_dependency[T: BaseModel](model_class: type[T]) -> Callable[..., T]:
    """Create a FastAPI dependency function from a Pydantic model for query parameters.

    FastAPI does not natively support using Pydantic models as query parameters
    without a dependency function. Using a dependency function typically leads to
    duplication: field types, default values, and descriptions must be repeated in
    `Query(...)` annotations even though they already exist in the Pydantic model.

    This function automatically generates a dependency function that extracts query parameters
    from the request and constructs an instance of the Pydantic model. The descriptions and
    defaults are automatically extracted from the model's Field definitions, making the model
    the single source of truth.

    Args:
        model_class: The Pydantic model class to create a dependency for

    Returns:
        A dependency function that can be used with FastAPI's Depends()
        ```
    """
    # Build function signature dynamically from model fields
    annotations: dict[str, Any] = {}
    defaults: dict[str, Any] = {}

    for field_name, field_info in model_class.model_fields.items():
        # Extract description from Field
        description = field_info.description

        # Create Query annotation with description from model
        query_annotation = Query(description=description) if description else Query()

        # Create Annotated type with Query
        field_type = field_info.annotation
        annotations[field_name] = Annotated[field_type, query_annotation]

        # Set default value from model
        if field_info.default is not inspect.Parameter.empty:
            defaults[field_name] = field_info.default

    # Create the dependency function dynamically
    def dependency_func(**kwargs: Any) -> T:
        return model_class(**kwargs)

    # Set function signature
    sig_params = []
    for field_name, field_type in annotations.items():
        default = defaults.get(field_name, inspect.Parameter.empty)
        param = inspect.Parameter(
            field_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default,
            annotation=field_type,
        )
        sig_params.append(param)

    # These attributes are set dynamically at runtime. While mypy can't verify them statically,
    # they are standard Python function attributes that exist on all callable objects at runtime.
    # Setting them allows FastAPI to properly introspect the function signature for dependency injection.
    dependency_func.__signature__ = inspect.Signature(sig_params)  # type: ignore[attr-defined]
    dependency_func.__annotations__ = annotations
    dependency_func.__name__ = f"get_{model_class.__name__.lower()}_request"

    return dependency_func


def create_path_dependency[T: BaseModel](model_class: type[T]) -> Callable[..., T]:
    """Create a FastAPI dependency function from a Pydantic model for path parameters.

    FastAPI requires path parameters to be explicitly annotated with `Path()`. When using
    a Pydantic model that contains path parameters, you typically need a dependency function
    that extracts the path parameter and constructs the model. This leads to duplication:
    the parameter name, type, and description must be repeated in `Path(...)` annotations
    even though they already exist in the Pydantic model.

    This function automatically generates a dependency function that extracts path parameters
    from the request and constructs an instance of the Pydantic model. The descriptions are
    automatically extracted from the model's Field definitions, making the model the single
    source of truth.

    Args:
        model_class: The Pydantic model class to create a dependency for. The model should
            have exactly one field that represents the path parameter.

    Returns:
        A dependency function that can be used with FastAPI's Depends()
        ```
    """
    # Get the single field from the model (path parameter models typically have one field)
    if len(model_class.model_fields) != 1:
        raise ValueError(
            f"Path parameter model {model_class.__name__} must have exactly one field, "
            f"but has {len(model_class.model_fields)} fields"
        )

    field_name, field_info = next(iter(model_class.model_fields.items()))

    # Extract description from Field
    description = field_info.description

    # Create Path annotation with description from model
    path_annotation = Path(description=description) if description else Path()

    # Create Annotated type with Path
    field_type = field_info.annotation
    annotations: dict[str, Any] = {field_name: Annotated[field_type, path_annotation]}

    # Create the dependency function dynamically
    def dependency_func(**kwargs: Any) -> T:
        return model_class(**kwargs)

    # Set function signature
    param = inspect.Parameter(
        field_name,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=annotations[field_name],
    )

    # These attributes are set dynamically at runtime. While mypy can't verify them statically,
    # they are standard Python function attributes that exist on all callable objects at runtime.
    # Setting them allows FastAPI to properly introspect the function signature for dependency injection.
    dependency_func.__signature__ = inspect.Signature([param])  # type: ignore[attr-defined]
    dependency_func.__annotations__ = annotations
    dependency_func.__name__ = f"get_{model_class.__name__.lower()}_request"

    return dependency_func


def try_translate_to_http_exception(exc: Exception) -> HTTPException | None:
    """Try to translate an exception to an HTTPException.

    Returns an HTTPException for known exception types (HTTPException,
    ValueError, and exceptions with a ``status_code`` attribute such as
    OGXError subclasses).  Returns ``None`` for unknown types so
    the caller can decide whether to re-raise the original exception.

    The full ``translate_exception`` pipeline lives in ``ogx.core``
    which ``ogx_api`` must not import, so we duplicate the minimal
    logic required here.
    """
    if isinstance(exc, HTTPException):
        return exc
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return HTTPException(status_code=status_code, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc) or "Invalid value")
    return None


class ExceptionTranslatingRoute(APIRoute):
    """Route class that converts known exception types to HTTPException.

    ValueError and OGXError (which carry a ``status_code``) are
    translated to the appropriate HTTPException so that FastAPI's built-in
    handler returns a proper JSON error response.  All other exceptions
    are left untouched so they can propagate to the server's global
    ``Exception`` handler registered via ``app.exception_handler(Exception)``.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                resp: Response = await original(request)
            except Exception as exc:
                http_exc = try_translate_to_http_exception(exc)
                if http_exc is not None:
                    raise http_exc from exc
                raise
            return resp

        return handler
