# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar


class ExtraBodyField[T]:
    """
    Marker annotation for parameters that arrive via extra_body in the client SDK.

    These parameters:
    - Will NOT appear in the generated client SDK method signature
    - WILL be documented in OpenAPI spec under x-ogx-extra-body-params
    - MUST be passed via the extra_body parameter in client SDK calls
    - WILL be available in server-side method signature with proper typing

    Example:
        ```python
        async def create_openai_response(
            self,
            input: str,
            model: str,
            shields: Annotated[
                list[str] | None, ExtraBodyField("List of shields to apply")
            ] = None,
        ) -> ResponseObject:
            # shields is available here with proper typing
            if shields:
                print(f"Using shields: {shields}")
        ```

        Client usage:
        ```python
        client.responses.create(
            input="hello", model="llama-3", extra_body={"shields": ["shield-1"]}
        )
        ```
    """

    def __init__(self, description: str | None = None):
        self.description = description


SchemaSource = Literal["json_schema_type", "registered_schema", "dynamic_schema"]


@dataclass(frozen=True)
class SchemaInfo:
    """Metadata describing a schema entry exposed to OpenAPI generation."""

    name: str
    type: Any
    source: SchemaSource


_json_schema_types: dict[type, SchemaInfo] = {}


def json_schema_type[T](cls: type[T]) -> type[T]:
    """
    Decorator to mark a Pydantic model for top-level component registration.

    Models marked with this decorator will be registered as top-level components
    in the OpenAPI schema, while unmarked models will be inlined.

    This provides control over schema registration to avoid unnecessary indirection
    for simple one-off types while keeping complex reusable types as components.
    """
    schema_name = getattr(cls, "__name__", f"Anonymous_{id(cls)}")
    _json_schema_types.setdefault(cls, SchemaInfo(name=schema_name, type=cls, source="json_schema_type"))
    return cls


# Global registries for schemas discoverable by the generator
_registered_schemas: dict[Any, SchemaInfo] = {}
_dynamic_schema_types: dict[type, SchemaInfo] = {}


def register_schema[T](schema_type: T, name: str | None = None) -> T:
    """
    Register a schema type for top-level component registration.

    This replicates the behavior of strong_typing's register_schema function.
    It's used for union types and other complex types that should appear as
    top-level components in the OpenAPI schema.

    Args:
        schema_type: The type to register (e.g., union types, Annotated types)
        name: Optional name for the schema in the OpenAPI spec. If not provided,
              uses the type's __name__ or a generated name.
    """
    if name is None:
        name = getattr(schema_type, "__name__", f"Anonymous_{id(schema_type)}")

    # Store the registration information in a global registry
    # since union types don't allow setting attributes
    _registered_schemas[schema_type] = SchemaInfo(name=name, type=schema_type, source="registered_schema")

    return schema_type


def get_registered_schema_info(schema_type: Any) -> SchemaInfo | None:
    """Return the registration metadata for a schema type if present."""
    return _registered_schemas.get(schema_type)


def iter_registered_schema_types() -> Iterable[SchemaInfo]:
    """Iterate over all explicitly registered schema entries."""
    return tuple(_registered_schemas.values())


def iter_json_schema_types() -> Iterable[type]:
    """Iterate over all Pydantic models decorated with @json_schema_type."""
    return tuple(info.type for info in _json_schema_types.values())


def get_json_schema_type_info(schema_type: type) -> SchemaInfo | None:
    """Return the registration metadata for a @json_schema_type decorated model if present."""
    return _json_schema_types.get(schema_type)


def iter_dynamic_schema_types() -> Iterable[type]:
    """Iterate over dynamic models registered at generation time."""
    return tuple(info.type for info in _dynamic_schema_types.values())


def register_dynamic_schema_type(schema_type: type, name: str | None = None) -> type:
    """Register a dynamic model generated at runtime for schema inclusion."""
    schema_name = name if name is not None else getattr(schema_type, "__name__", f"Anonymous_{id(schema_type)}")
    _dynamic_schema_types[schema_type] = SchemaInfo(name=schema_name, type=schema_type, source="dynamic_schema")
    return schema_type


def clear_dynamic_schema_types() -> None:
    """Clear dynamic schema registrations."""
    _dynamic_schema_types.clear()


@dataclass
class WebMethod:
    """Metadata container for an API endpoint operation's routing and configuration."""

    level: str | None = None
    route: str | None = None
    public: bool = False
    request_examples: list[Any] | None = None
    response_examples: list[Any] | None = None
    method: str | None = None
    raw_bytes_request_body: bool | None = False
    # A descriptive name of the corresponding span created by tracing
    descriptive_name: str | None = None
    deprecated: bool | None = False
    require_authentication: bool | None = True


CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def webmethod(
    route: str | None = None,
    method: str | None = None,
    level: str | None = None,
    public: bool | None = False,
    request_examples: list[Any] | None = None,
    response_examples: list[Any] | None = None,
    raw_bytes_request_body: bool | None = False,
    descriptive_name: str | None = None,
    deprecated: bool | None = False,
    require_authentication: bool | None = True,
) -> Callable[[CallableT], CallableT]:
    """
    Decorator that supplies additional metadata to an endpoint operation function.

    :param route: The URL path pattern associated with this operation which path parameters are substituted into.
    :param public: True if the operation can be invoked without prior authentication.
    :param request_examples: Sample requests that the operation might take. Pass a list of objects, not JSON.
    :param response_examples: Sample responses that the operation might produce. Pass a list of objects, not JSON.
    :param require_authentication: Whether this endpoint requires authentication (default True).
    """

    def wrap(func: CallableT) -> CallableT:
        webmethod_obj = WebMethod(
            route=route,
            method=method,
            level=level,
            public=public or False,
            request_examples=request_examples,
            response_examples=response_examples,
            raw_bytes_request_body=raw_bytes_request_body,
            descriptive_name=descriptive_name,
            deprecated=deprecated,
            require_authentication=require_authentication if require_authentication is not None else True,
        )

        # Store all webmethods in a list to support multiple decorators
        if not hasattr(func, "__webmethods__"):
            func.__webmethods__ = []  # type: ignore
        func.__webmethods__.append(webmethod_obj)  # type: ignore

        # Keep the last one as __webmethod__ for backwards compatibility
        func.__webmethod__ = webmethod_obj  # type: ignore
        return func

    return wrap


def remove_null_from_anyof(schema: dict[str, Any], *, add_nullable: bool = False) -> None:
    """Remove null type from anyOf and optionally add nullable flag.

    Converts Pydantic's default OpenAPI 3.1 style:
        anyOf: [{type: X, enum: [...]}, {type: null}]

    To flattened format:
        type: X
        enum: [...]
        nullable: true  # only if add_nullable=True

    Args:
        schema: The JSON schema dict to modify in-place
        add_nullable: If True, adds 'nullable: true' when null was present.
                      Use True for OpenAPI 3.0 compatibility with OpenAI's spec.
    """
    # Handle anyOf format: anyOf: [{type: string, enum: [...]}, {type: null}]
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        has_null = len(non_null) < len(schema["anyOf"])

        if len(non_null) == 1:
            # Flatten to single type
            only_schema = non_null[0]
            schema.pop("anyOf")
            schema.update(only_schema)
            if has_null and add_nullable:
                schema["nullable"] = True

    # Handle OpenAPI 3.1 format: type: ['string', 'null']
    elif isinstance(schema.get("type"), list) and "null" in schema["type"]:
        has_null = "null" in schema["type"]
        schema["type"].remove("null")
        if len(schema["type"]) == 1:
            schema["type"] = schema["type"][0]
        if has_null and add_nullable:
            schema["nullable"] = True


def nullable_openai_style(schema: dict[str, Any]) -> None:
    """Shorthand for remove_null_from_anyof with add_nullable=True.

    Use this for fields that need OpenAPI 3.0 nullable style to match OpenAI's spec.
    """
    remove_null_from_anyof(schema, add_nullable=True)
