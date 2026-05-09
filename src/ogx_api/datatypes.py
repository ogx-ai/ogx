# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Iterator
from enum import Enum, EnumMeta, StrEnum
from typing import Any, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ogx_api.models import Model
from ogx_api.schema_utils import json_schema_type
from ogx_api.shields import Shield
from ogx_api.tools import ToolGroup
from ogx_api.vector_stores import VectorStore


class DynamicApiMeta(EnumMeta):
    """Metaclass that allows dynamic addition of enum members at runtime."""

    def __new__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> type:
        # Store the original enum values
        original_values = {k: v for k, v in namespace.items() if not k.startswith("_")}

        # Create the enum class
        enum_cls = super().__new__(cls, name, bases, namespace)  # type: ignore[arg-type]

        # Store the original values for reference
        enum_cls._original_values = original_values  # type: ignore[attr-defined]
        # Initialize _dynamic_values
        enum_cls._dynamic_values: dict[str, Any] = {}  # type: ignore[attr-defined,misc]

        return enum_cls

    def __call__(cls, value: Any) -> Any:  # type: ignore[override]
        try:
            return super().__call__(value)
        except ValueError as e:
            # If this value was already dynamically added, return it
            if value in cls._dynamic_values:  # type: ignore[attr-defined]
                return cls._dynamic_values[value]  # type: ignore[attr-defined]

            # If the value doesn't exist, create a new enum member
            # Create a new member name from the value
            member_name = value.lower().replace("-", "_")

            # If this member name already exists in the enum, return the existing member
            if member_name in cls._member_map_:
                return cls._member_map_[member_name]

            # Instead of creating a new member, raise ValueError to force users to use Api.add() to
            # register new APIs explicitly
            raise ValueError(f"API '{value}' does not exist. Use Api.add() to register new APIs.") from e

    def __iter__(cls) -> Iterator[Any]:
        # Allow iteration over both static and dynamic members
        yield from super().__iter__()
        if hasattr(cls, "_dynamic_values"):
            yield from cls._dynamic_values.values()

    def add(cls, value: str) -> Any:
        """
        Add a new API to the enum.
        Used to register external APIs.
        """
        member_name = value.lower().replace("-", "_")

        # If this member name already exists in the enum, return it
        if member_name in cls._member_map_:
            return cls._member_map_[member_name]

        # Create a new enum member
        member: Any = object.__new__(cls)  # type: ignore[arg-type]
        member._name_ = member_name
        member._value_ = value

        # Add it to the enum class
        cls._member_map_[member_name] = member
        cls._member_names_.append(member_name)
        cls._member_type_ = str

        # Store it in our dynamic values
        cls._dynamic_values[value] = member  # type: ignore[attr-defined]

        return member


@json_schema_type
class Api(Enum, metaclass=DynamicApiMeta):
    """Enumeration of all available APIs in the OGX system.
    :cvar providers: Provider management and configuration
    :cvar inference: Text generation, chat completions, and embeddings
    :cvar safety: Content moderation and safety shields
    :cvar responses: Response orchestration and execution
    :cvar batches: Batch processing for asynchronous API requests
    :cvar vector_io: Vector database operations and queries
    :cvar tool_runtime: Tool execution and management
    :cvar telemetry: Observability and system monitoring
    :cvar models: Model metadata and management
    :cvar shields: Safety shield implementations
    :cvar tool_groups: Tool group organization
    :cvar files: File storage and management
    :cvar file_processors: File parsing and processing operations
    :cvar prompts: Prompt versions and management
    :cvar connectors: External connector management (e.g., MCP servers)
    :cvar messages: Anthropic Messages API compatibility layer
    :cvar interactions: Google Interactions API compatibility layer
    :cvar inspect: Built-in system inspection and introspection
    """

    providers = "providers"
    inference = "inference"
    safety = "safety"
    responses = "responses"
    batches = "batches"
    vector_io = "vector_io"
    tool_runtime = "tool_runtime"

    models = "models"
    shields = "shields"
    vector_stores = "vector_stores"  # only used for routing table
    tool_groups = "tool_groups"
    files = "files"
    file_processors = "file_processors"
    prompts = "prompts"
    conversations = "conversations"
    connectors = "connectors"
    messages = "messages"
    interactions = "interactions"

    # built-in API
    inspect = "inspect"
    admin = "admin"


@json_schema_type
class Error(BaseModel):
    """
    Error response from the API. Roughly follows RFC 7807.

    :param status: HTTP status code
    :param title: Error title, a short summary of the error which is invariant for an error type
    :param detail: Error detail, a longer human-readable description of the error
    :param instance: (Optional) A URL which can be used to retrieve more information about the specific occurrence of the error
    """

    status: int
    title: str
    detail: str
    instance: str | None = None


class ExternalApiSpec(BaseModel):
    """Specification for an external API implementation."""

    module: str = Field(..., description="Python module containing the API implementation")
    name: str = Field(..., description="Name of the API")
    pip_packages: list[str] = Field(default=[], description="List of pip packages to install the API")
    protocol: str = Field(..., description="Name of the protocol class for the API")


# Provider-related types (merged from providers/datatypes.py)
# NOTE: These imports are forward references to avoid circular dependencies
# They will be resolved at runtime when the classes are used


class ModelsProtocolPrivate(Protocol):
    """
    Protocol for model management.

    This allows users to register their preferred model identifiers.

    Model registration requires -
     - a provider, used to route the registration request
     - a model identifier, user's intended name for the model during inference
     - a provider model identifier, a model identifier supported by the provider

    Providers will only accept registration for provider model ids they support.

    Example,
      register: provider x my-model-id x provider-model-id
       -> Error if provider does not support provider-model-id
       -> Error if my-model-id is already registered
       -> Success if provider supports provider-model-id
      inference: my-model-id x ...
       -> Provider uses provider-model-id for inference
    """

    # this should be called `on_model_register` or something like that.
    # the provider should _not_ be able to change the object in this
    # callback
    async def register_model(self, model: Model) -> Model: ...

    async def unregister_model(self, model_id: str) -> None: ...

    # the Stack router will query each provider for their list of models
    # if a `refresh_interval_seconds` is provided, this method will be called
    # periodically to refresh the list of models
    #
    # NOTE: each model returned will be registered with the model registry. this means
    # a callback to the `register_model()` method will be made. this is duplicative and
    # may be removed in the future.
    async def list_models(self) -> list[Model] | None: ...

    async def should_refresh_models(self) -> bool: ...


class ShieldsProtocolPrivate(Protocol):
    """Protocol for provider-side shield registration and unregistration."""

    async def register_shield(self, shield: Shield) -> None: ...

    async def unregister_shield(self, identifier: str) -> None: ...


class VectorStoresProtocolPrivate(Protocol):
    """Protocol for provider-side vector store registration and unregistration."""

    async def register_vector_store(self, vector_store: VectorStore) -> None: ...

    async def unregister_vector_store(self, vector_store_id: str) -> None: ...


class ToolGroupsProtocolPrivate(Protocol):
    """Protocol for provider-side tool group registration and unregistration."""

    async def register_toolgroup(self, toolgroup: ToolGroup) -> None: ...

    async def unregister_toolgroup(self, toolgroup_id: str) -> None: ...


@json_schema_type
class ProviderSpec(BaseModel):
    """Base specification for a OGX provider including its config and dependencies."""

    api: Api
    provider_type: str
    config_class: str = Field(
        ...,
        description="Fully-qualified classname of the config for this provider",
    )
    api_dependencies: list[Api] = Field(
        default_factory=list,
        description="Higher-level API surfaces may depend on other providers to provide their functionality",
    )
    optional_api_dependencies: list[Api] = Field(
        default_factory=list,
    )
    deprecation_warning: str | None = Field(
        default=None,
        description="If this provider is deprecated, specify the warning message here",
    )
    deprecation_error: str | None = Field(
        default=None,
        description="If this provider is deprecated and does NOT work, specify the error message here",
    )

    module: str | None = Field(
        default=None,
        description="""
 Fully-qualified name of the module to import. The module is expected to have:

  - `get_adapter_impl(config, deps)`: returns the adapter implementation

  Example: `module: ramalama_stack`
 """,
    )

    pip_packages: list[str] = Field(
        default_factory=list,
        description="The pip dependencies needed for this implementation",
    )

    provider_data_validator: str | None = Field(
        default=None,
    )

    is_external: bool = Field(
        default=False,
        description="Notes whether this provider is an external provider.",
    )

    toolgroup_id: str | None = Field(
        default=None,
        description="For tool_runtime providers, the built-in tool group ID this provider serves (e.g. 'builtin::websearch').",
    )

    # used internally by the resolver; this is a hack for now
    deps__: list[str] = Field(default_factory=list)

    @property
    def is_sample(self) -> bool:
        return self.provider_type in ("sample", "remote::sample")


class RoutingTable(Protocol):
    """Protocol for resolving a routing key to its provider implementation."""

    async def get_provider_impl(self, routing_key: str) -> Any: ...


@json_schema_type
class InlineProviderSpec(ProviderSpec):
    """Provider specification for inline (built-in) providers with optional container support."""

    container_image: str | None = Field(
        default=None,
        description="""
The container image to use for this implementation. If one is provided, pip_packages will be ignored.
If a provider depends on other providers, the dependencies MUST NOT specify a container image.
""",
    )
    description: str | None = Field(
        default=None,
        description="""
A description of the provider. This is used to display in the documentation.
""",
    )


class RemoteProviderConfig(BaseModel):
    """Connection configuration for a remote provider endpoint."""

    host: str = "localhost"
    port: int | None = None
    protocol: str = "http"

    @property
    def url(self) -> str:
        if self.port is None:
            return f"{self.protocol}://{self.host}"
        return f"{self.protocol}://{self.host}:{self.port}"

    @classmethod
    def from_url(cls, url: str) -> "RemoteProviderConfig":
        parsed = urlparse(url)
        attrs = {k: v for k, v in parsed._asdict().items() if v is not None}
        return cls(**attrs)


@json_schema_type
class RemoteProviderSpec(ProviderSpec):
    """Provider specification for remote providers accessed via an adapter."""

    adapter_type: str = Field(
        ...,
        description="Unique identifier for this adapter",
    )

    description: str | None = Field(
        default=None,
        description="""
A description of the provider. This is used to display in the documentation.
""",
    )

    @property
    def container_image(self) -> str | None:
        return None


class HealthStatus(StrEnum):
    """Health check status values for provider readiness."""

    OK = "OK"
    ERROR = "Error"
    NOT_IMPLEMENTED = "Not Implemented"


HealthResponse = dict[str, Any]
