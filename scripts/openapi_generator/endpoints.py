# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Endpoint generation logic for FastAPI OpenAPI generation.
"""

import inspect
import re
import types
import typing
from typing import Annotated, Any, get_args, get_origin

from fastapi import FastAPI
from fastapi.params import Body as FastAPIBody
from pydantic import Field, create_model

from ogx.log import get_logger
from ogx_api import Api
from ogx_api.schema_utils import get_registered_schema_info

from . import app as app_module
from .state import _extra_body_fields, register_dynamic_model

logger = get_logger(name=__name__, category="core")

type QueryParameter = tuple[str, type, Any, bool]


def _to_pascal_case(segment: str) -> str:
    tokens = re.findall(r"[A-Za-z]+|\d+", segment)
    return "".join(token.capitalize() for token in tokens if token)


def _compose_request_model_name(api: Api, method_name: str, variant: str | None = None) -> str:
    """Generate a deterministic model name from the protocol method."""

    def _to_pascal_from_snake(value: str) -> str:
        return "".join(segment.capitalize() for segment in value.split("_") if segment)

    base_name = _to_pascal_from_snake(method_name)
    if not base_name:
        base_name = _to_pascal_case(api.value)
    base_name = f"{base_name}Request"
    if variant:
        base_name = f"{base_name}{variant}"
    return base_name


def _extract_path_parameters(path: str) -> list[dict[str, Any]]:
    """Extract path parameters from a URL path and return them as OpenAPI parameter definitions."""
    matches = re.findall(r"\{([^}:]+)(?::[^}]+)?\}", path)
    return [
        {
            "name": param_name,
            "in": "path",
            "required": True,
            "schema": {"type": "string"},
            "description": f"Path parameter: {param_name}",
        }
        for param_name in matches
    ]


def _create_endpoint_with_request_model(
    request_model: type, response_model: type | None, operation_description: str | None
):
    """Create an endpoint function with a request body model."""

    async def endpoint(request: request_model) -> response_model:
        return response_model() if response_model else {}

    if operation_description:
        endpoint.__doc__ = operation_description
    return endpoint


def _build_field_definitions(query_parameters: list[QueryParameter], use_any: bool = False) -> dict[str, tuple]:
    """Build field definitions for a Pydantic model from query parameters."""
    from typing import Any

    field_definitions = {}
    for param_name, param_type, default_value, _ in query_parameters:
        if use_any:
            field_definitions[param_name] = (Any, ... if default_value is inspect.Parameter.empty else default_value)
            continue

        base_type = param_type
        extracted_field = None
        if get_origin(param_type) is Annotated:
            args = get_args(param_type)
            if args:
                base_type = args[0]
                for arg in args[1:]:
                    if isinstance(arg, Field):
                        extracted_field = arg
                        break

        try:
            if extracted_field:
                field_definitions[param_name] = (base_type, extracted_field)
            else:
                field_definitions[param_name] = (
                    base_type,
                    ... if default_value is inspect.Parameter.empty else default_value,
                )
        except (TypeError, ValueError):
            field_definitions[param_name] = (Any, ... if default_value is inspect.Parameter.empty else default_value)

    # Ensure all parameters are included
    expected_params = {name for name, _, _, _ in query_parameters}
    missing = expected_params - set(field_definitions.keys())
    if missing:
        for param_name, _, default_value, _ in query_parameters:
            if param_name in missing:
                field_definitions[param_name] = (
                    Any,
                    ... if default_value is inspect.Parameter.empty else default_value,
                )

    return field_definitions


def _create_dynamic_request_model(
    api: Api,
    webmethod,
    method_name: str,
    http_method: str,
    query_parameters: list[QueryParameter],
    use_any: bool = False,
    variant_suffix: str | None = None,
) -> type | None:
    """Create a dynamic Pydantic model for request body."""
    try:
        field_definitions = _build_field_definitions(query_parameters, use_any)
        if not field_definitions:
            return None
        model_name = _compose_request_model_name(api, method_name, variant_suffix or None)
        request_model = create_model(model_name, **field_definitions)
        return register_dynamic_model(model_name, request_model)
    except Exception:
        return None


def _build_signature_params(
    query_parameters: list[QueryParameter],
) -> tuple[list[inspect.Parameter], dict[str, type]]:
    """Build signature parameters and annotations from query parameters."""
    signature_params = []
    param_annotations = {}
    for param_name, param_type, default_value, _ in query_parameters:
        param_annotations[param_name] = param_type
        signature_params.append(
            inspect.Parameter(
                param_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default_value if default_value is not inspect.Parameter.empty else inspect.Parameter.empty,
                annotation=param_type,
            )
        )
    return signature_params, param_annotations


def _extract_operation_description_from_docstring(api: Api, method_name: str) -> str | None:
    """Extract operation description from the actual function docstring."""
    func = app_module._get_protocol_method(api, method_name)
    if not func or not func.__doc__:
        return None

    doc_lines = func.__doc__.split("\n")
    description_lines = []
    metadata_markers = (":param", ":type", ":return", ":returns", ":raises", ":exception", ":yield", ":yields", ":cvar")

    for line in doc_lines:
        if line.strip().startswith(metadata_markers):
            break
        description_lines.append(line)

    description = "\n".join(description_lines).strip()
    return description if description else None


def _extract_response_description_from_docstring(webmethod, response_model, api: Api, method_name: str) -> str:
    """Extract response description from the actual function docstring."""
    func = app_module._get_protocol_method(api, method_name)
    if not func or not func.__doc__:
        return "Successful Response"
    for line in func.__doc__.split("\n"):
        if line.strip().startswith(":returns:"):
            if desc := line.strip()[9:].strip():
                return desc
    return "Successful Response"


def _get_tag_from_api(api: Api) -> str:
    """Extract a tag name from the API enum for API grouping."""
    return api.value.replace("_", " ").title()


def _is_file_or_form_param(param_type: Any) -> bool:
    """Check if a parameter type is annotated with File() or Form()."""
    if get_origin(param_type) is Annotated:
        args = get_args(param_type)
        if len(args) > 1:
            # Check metadata for File or Form
            for metadata in args[1:]:
                # Check if it's a File or Form instance
                if hasattr(metadata, "__class__"):
                    class_name = metadata.__class__.__name__
                    if class_name in ("File", "Form"):
                        return True
    return False


def _is_extra_body_field(metadata_item: Any) -> bool:
    """Check if a metadata item is an ExtraBodyField instance."""
    from ogx_api.schema_utils import ExtraBodyField

    return isinstance(metadata_item, ExtraBodyField)


def _should_embed_parameter(param_type: Any) -> bool:
    """Determine whether a parameter should be embedded (wrapped) in the request body."""
    if get_origin(param_type) is Annotated:
        args = get_args(param_type)
        metadata = args[1:] if len(args) > 1 else []
        for metadata_item in metadata:
            if isinstance(metadata_item, FastAPIBody):
                # FastAPI treats embed=None as False, so default to False when unset.
                return bool(metadata_item.embed)
    # Unannotated parameters default to embed=True through create_dynamic_typed_route.
    return True


def _is_async_iterator_type(type_obj: Any) -> bool:
    """Check if a type is AsyncIterator or AsyncIterable."""
    from collections.abc import AsyncIterable, AsyncIterator

    origin = get_origin(type_obj)
    if origin is None:
        # Check if it's the class itself
        return type_obj in (AsyncIterator, AsyncIterable) or (
            hasattr(type_obj, "__origin__") and type_obj.__origin__ in (AsyncIterator, AsyncIterable)
        )
    return origin in (AsyncIterator, AsyncIterable)


def _extract_response_models_from_union(union_type: Any) -> tuple[type | None, type | None]:
    """
    Extract non-streaming and streaming response models from a union type.

    Returns:
        tuple: (non_streaming_model, streaming_model)
    """
    non_streaming_model = None
    streaming_model = None

    args = get_args(union_type)
    for arg in args:
        # Check if it's an AsyncIterator
        if _is_async_iterator_type(arg):
            # Extract the type argument from AsyncIterator[T]
            iterator_args = get_args(arg)
            if iterator_args:
                inner_type = iterator_args[0]
                # Check if the inner type is a registered schema (union type)
                # or a Pydantic model
                if hasattr(inner_type, "model_json_schema"):
                    streaming_model = inner_type
                else:
                    # Might be a registered schema - check if it's registered
                    if get_registered_schema_info(inner_type):
                        # We'll need to look this up later, but for now store the type
                        streaming_model = inner_type
        elif hasattr(arg, "model_json_schema"):
            # Non-streaming Pydantic model
            if non_streaming_model is None:
                non_streaming_model = arg

    return non_streaming_model, streaming_model


def _find_models_for_endpoint(
    webmethod, api: Api, method_name: str, is_post_put: bool = False
) -> tuple[type | None, type | None, list[tuple[str, type, Any]], list[inspect.Parameter], type | None, str | None]:
    """
    Find appropriate request and response models for an endpoint by analyzing the actual function signature.
    This uses the protocol function to determine the correct models dynamically.

    Args:
        webmethod: The webmethod metadata
        api: The API enum for looking up the function
        method_name: The method name (function name)
        is_post_put: Whether this is a POST, PUT, or PATCH request (GET requests should never have request bodies)

    Returns:
        tuple: (request_model, response_model, query_parameters, file_form_params, streaming_response_model, response_schema_name)
        where query_parameters is a list of (name, type, default_value, should_embed) tuples
        and file_form_params is a list of inspect.Parameter objects for File()/Form() params
        and streaming_response_model is the model for streaming responses (AsyncIterator content)
    """
    route_descriptor = f"{webmethod.method or 'UNKNOWN'} {webmethod.route}"
    try:
        # Get the function from the protocol
        func = app_module._get_protocol_method(api, method_name)
        if not func:
            logger.warning("No protocol method for %s.%s (%s)", api, method_name, route_descriptor)
            return None, None, [], [], None, None

        # Analyze the function signature
        sig = inspect.signature(func)

        # Find request model and collect all body parameters
        request_model = None
        query_parameters: list[QueryParameter] = []
        file_form_params = []
        path_params = set()
        extra_body_params = []
        response_schema_name = None

        # Extract path parameters from the route
        if webmethod and hasattr(webmethod, "route"):
            path_matches = re.findall(r"\{([^}:]+)(?::[^}]+)?\}", webmethod.route)
            path_params = set(path_matches)

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Skip *args and **kwargs parameters - these are not real API parameters
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            # Check if this is a path parameter
            if param_name in path_params:
                # Path parameters are handled separately, skip them
                continue

            # Check if it's a File() or Form() parameter - these need special handling
            param_type = param.annotation
            param_should_embed = _should_embed_parameter(param_type)
            if _is_file_or_form_param(param_type):
                # File() and Form() parameters must be in the function signature directly
                # They cannot be part of a Pydantic model
                file_form_params.append(param)
                continue

            # Check for ExtraBodyField in Annotated types
            is_extra_body = False
            extra_body_description = None
            if get_origin(param_type) is Annotated:
                args = get_args(param_type)
                base_type = args[0] if args else param_type
                metadata = args[1:] if len(args) > 1 else []

                # Check if any metadata item is an ExtraBodyField
                for metadata_item in metadata:
                    if _is_extra_body_field(metadata_item):
                        is_extra_body = True
                        extra_body_description = metadata_item.description
                        break

                if is_extra_body:
                    # Store as extra body parameter - exclude from request model
                    extra_body_params.append((param_name, base_type, extra_body_description))
                    continue
                param_type = base_type

            # Check if it's a Pydantic model (for POST/PUT requests)
            if hasattr(param_type, "model_json_schema"):
                query_parameters.append((param_name, param_type, param.default, param_should_embed))
            else:
                # Regular annotated parameter (but not File/Form, already handled above)
                query_parameters.append((param_name, param_type, param.default, param_should_embed))

        # Store extra body fields for later use in post-processing
        # We'll store them when the endpoint is created, as we need the full path
        # For now, attach to the function for later retrieval
        if extra_body_params:
            func._extra_body_params = extra_body_params  # type: ignore

        # If there's exactly one body parameter and it's a Pydantic model, use it directly
        # Otherwise, we'll create a combined request model from all parameters
        # BUT: For GET requests, never create a request body - all parameters should be query parameters
        if is_post_put and len(query_parameters) == 1:
            param_name, param_type, default_value, should_embed = query_parameters[0]
            if hasattr(param_type, "model_json_schema") and not should_embed:
                request_model = param_type
                query_parameters = []  # Clear query_parameters so we use the single model

        # Find response model from return annotation
        # Also detect streaming response models (AsyncIterator)
        response_model = None
        streaming_response_model = None
        return_annotation = sig.return_annotation
        if return_annotation != inspect.Signature.empty:
            origin = get_origin(return_annotation)
            if hasattr(return_annotation, "model_json_schema"):
                response_model = return_annotation
            elif origin is Annotated:
                # Handle Annotated return types
                args = get_args(return_annotation)
                if args:
                    # Check if the first argument is a Pydantic model
                    if hasattr(args[0], "model_json_schema"):
                        response_model = args[0]
                    else:
                        # Check if the first argument is a union type
                        inner_origin = get_origin(args[0])
                        if inner_origin is not None and (
                            inner_origin is types.UnionType or inner_origin is typing.Union
                        ):
                            response_model, streaming_response_model = _extract_response_models_from_union(args[0])
            elif origin is not None and (origin is types.UnionType or origin is typing.Union):
                # Handle union types - extract both non-streaming and streaming models
                response_model, streaming_response_model = _extract_response_models_from_union(return_annotation)
            else:
                try:
                    from fastapi import Response as FastAPIResponse
                except ImportError:
                    fastapi_response_cls = None
                else:
                    fastapi_response_cls = FastAPIResponse
                try:
                    from starlette.responses import Response as StarletteResponse
                except ImportError:
                    starlette_response_cls = None
                else:
                    starlette_response_cls = StarletteResponse

                response_types = tuple(t for t in (fastapi_response_cls, starlette_response_cls) if t is not None)
                if response_types and any(return_annotation is t for t in response_types):
                    response_schema_name = "Response"

        return (
            request_model,
            response_model,
            query_parameters,
            file_form_params,
            streaming_response_model,
            response_schema_name,
        )

    except Exception as exc:
        logger.warning(
            "Failed to analyze endpoint %s.%s (%s): %s", api, method_name, route_descriptor, exc, exc_info=True
        )
        return None, None, [], [], None, None


def _create_fastapi_endpoint(app: FastAPI, route, webmethod, api: Api):
    """Create a FastAPI endpoint from a discovered route and webmethod."""
    path = route.path
    raw_methods = route.methods or set()
    method_list = sorted({method.upper() for method in raw_methods if method and method.upper() != "HEAD"})
    if not method_list:
        method_list = ["GET"]
    primary_method = method_list[0]
    name = route.name
    fastapi_path = path.replace("{", "{").replace("}", "}")
    is_post_put = any(method in ["POST", "PUT", "PATCH"] for method in method_list)

    (
        request_model,
        response_model,
        query_parameters,
        file_form_params,
        streaming_response_model,
        response_schema_name,
    ) = _find_models_for_endpoint(webmethod, api, name, is_post_put)
    operation_description = _extract_operation_description_from_docstring(api, name)
    response_description = _extract_response_description_from_docstring(webmethod, response_model, api, name)

    # Retrieve and store extra body fields for this endpoint
    func = app_module._get_protocol_method(api, name)
    extra_body_params = getattr(func, "_extra_body_params", []) if func else []
    if extra_body_params:
        for method in method_list:
            key = (fastapi_path, method.upper())
            _extra_body_fields[key] = extra_body_params

    if is_post_put and not request_model and not file_form_params and query_parameters:
        request_model = _create_dynamic_request_model(
            api, webmethod, name, primary_method, query_parameters, use_any=False
        )
        if not request_model:
            request_model = _create_dynamic_request_model(
                api, webmethod, name, primary_method, query_parameters, use_any=True, variant_suffix="Loose"
            )
        if request_model:
            query_parameters = []

    if file_form_params and is_post_put:
        signature_params = list(file_form_params)
        param_annotations = {param.name: param.annotation for param in file_form_params}
        for param_name, param_type, default_value, _ in query_parameters:
            signature_params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default_value if default_value is not inspect.Parameter.empty else inspect.Parameter.empty,
                    annotation=param_type,
                )
            )
            param_annotations[param_name] = param_type

        async def file_form_endpoint():
            return response_model() if response_model else {}

        if operation_description:
            file_form_endpoint.__doc__ = operation_description
        file_form_endpoint.__signature__ = inspect.Signature(signature_params)
        file_form_endpoint.__annotations__ = param_annotations
        endpoint_func = file_form_endpoint
    elif request_model and response_model:
        endpoint_func = _create_endpoint_with_request_model(request_model, response_model, operation_description)
    elif request_model:
        endpoint_func = _create_endpoint_with_request_model(request_model, None, operation_description)
    elif response_model and query_parameters:
        if is_post_put:
            request_model = _create_dynamic_request_model(
                api, webmethod, name, primary_method, query_parameters, use_any=False
            )
            if not request_model:
                request_model = _create_dynamic_request_model(
                    api, webmethod, name, primary_method, query_parameters, use_any=True, variant_suffix="Loose"
                )

            if request_model:
                endpoint_func = _create_endpoint_with_request_model(
                    request_model, response_model, operation_description
                )
            else:

                async def empty_endpoint() -> response_model:
                    return response_model() if response_model else {}

                if operation_description:
                    empty_endpoint.__doc__ = operation_description
                endpoint_func = empty_endpoint
        else:
            sorted_params = sorted(query_parameters, key=lambda x: (x[2] is not inspect.Parameter.empty, x[0]))
            signature_params, param_annotations = _build_signature_params(sorted_params)

            async def query_endpoint():
                return response_model()

            if operation_description:
                query_endpoint.__doc__ = operation_description
            query_endpoint.__signature__ = inspect.Signature(signature_params)
            query_endpoint.__annotations__ = param_annotations
            endpoint_func = query_endpoint
    elif response_model:

        async def response_only_endpoint() -> response_model:
            return response_model()

        if operation_description:
            response_only_endpoint.__doc__ = operation_description
        endpoint_func = response_only_endpoint
    elif query_parameters:
        signature_params, param_annotations = _build_signature_params(query_parameters)

        async def params_only_endpoint():
            return {}

        if operation_description:
            params_only_endpoint.__doc__ = operation_description
        params_only_endpoint.__signature__ = inspect.Signature(signature_params)
        params_only_endpoint.__annotations__ = param_annotations
        endpoint_func = params_only_endpoint
    else:
        # Endpoint with no parameters and no response model
        # If we have a response_model from the function signature, use it even if _find_models_for_endpoint didn't find it
        # This can happen if there was an exception during model finding
        if response_model is None:
            # Try to get response model directly from the function signature as a fallback
            func = app_module._get_protocol_method(api, name)
            if func:
                try:
                    sig = inspect.signature(func)
                    return_annotation = sig.return_annotation
                    if return_annotation != inspect.Signature.empty:
                        if hasattr(return_annotation, "model_json_schema"):
                            response_model = return_annotation
                        elif get_origin(return_annotation) is Annotated:
                            args = get_args(return_annotation)
                            if args and hasattr(args[0], "model_json_schema"):
                                response_model = args[0]
                except Exception:
                    pass

        if response_model:

            async def no_params_endpoint() -> response_model:
                return response_model() if response_model else {}
        else:

            async def no_params_endpoint():
                return {}

        if operation_description:
            no_params_endpoint.__doc__ = operation_description
        endpoint_func = no_params_endpoint

    # Build response content with both application/json and text/event-stream if streaming
    response_content: dict[str, Any] = {}
    if response_model:
        response_content["application/json"] = {"schema": {"$ref": f"#/components/schemas/{response_model.__name__}"}}
    elif response_schema_name:
        response_content["application/json"] = {"schema": {"$ref": f"#/components/schemas/{response_schema_name}"}}
    if streaming_response_model:
        # Get the schema name for the streaming model
        # It might be a registered schema or a Pydantic model
        streaming_schema_name = None
        # Check if it's a registered schema first (before checking __name__)
        # because registered schemas might be Annotated types
        if schema_info := get_registered_schema_info(streaming_response_model):
            streaming_schema_name = schema_info.name
        elif hasattr(streaming_response_model, "__name__"):
            streaming_schema_name = streaming_response_model.__name__

        if streaming_schema_name:
            response_content["text/event-stream"] = {
                "schema": {"$ref": f"#/components/schemas/{streaming_schema_name}"}
            }

    # If no content types, use empty schema
    # Add the endpoint to the FastAPI app
    is_deprecated = webmethod.deprecated or False
    route_kwargs = {
        "name": name,
        "tags": [_get_tag_from_api(api)],
        "deprecated": is_deprecated,
        "responses": {
            400: {"$ref": "#/components/responses/BadRequest400"},
            429: {"$ref": "#/components/responses/TooManyRequests429"},
            500: {"$ref": "#/components/responses/InternalServerError500"},
            "default": {"$ref": "#/components/responses/DefaultError"},
        },
    }
    success_response: dict[str, Any] = {"description": response_description}
    if response_content:
        success_response["content"] = response_content
    route_kwargs["responses"][200] = success_response

    # FastAPI needs response_model parameter to properly generate OpenAPI spec
    # Use the non-streaming response model if available
    if response_model:
        route_kwargs["response_model"] = response_model

    method_map = {"GET": app.get, "POST": app.post, "PUT": app.put, "DELETE": app.delete, "PATCH": app.patch}
    for method in method_list:
        if handler := method_map.get(method):
            handler(fastapi_path, **route_kwargs)(endpoint_func)
