# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import warnings
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from ogx.core.access_control.datatypes import AccessRule, RouteAccessRule
from ogx.core.storage.datatypes import (
    KVStoreReference,
    StorageBackendType,
    StorageConfig,
)
from ogx.log import LoggingConfig
from ogx_api import (
    Api,
    ConnectorInput,
    Inference,
    Model,
    ModelInput,
    ProviderSpec,
    Resource,
    Safety,
    Shield,
    ShieldInput,
    ToolGroup,
    ToolGroupInput,
    ToolRuntime,
    VectorIO,
    VectorStore,
    VectorStoreInput,
)

OGX_BUILD_CONFIG_VERSION = 2
OGX_RUN_CONFIG_VERSION = 2


RoutingKey = str | list[str]


class RegistryEntrySource(StrEnum):
    """Source of a registry entry, distinguishing user-registered from provider-listed resources."""

    via_register_api = "via_register_api"
    listed_from_provider = "listed_from_provider"


class User(BaseModel):
    """An authenticated user with a principal identity and optional access control attributes."""

    principal: str
    # further attributes that may be used for access control decisions
    attributes: dict[str, list[str]] | None = None

    def __init__(self, principal: str, attributes: dict[str, list[str]] | None):
        super().__init__(principal=principal, attributes=attributes)


class ResourceWithOwner(Resource):
    """Extension of Resource that adds an optional owner, i.e. the user that created the
    resource. This can be used to constrain access to the resource."""

    owner: User | None = None
    source: RegistryEntrySource = RegistryEntrySource.via_register_api


# Use the extended Resource for all routable objects
class ModelWithOwner(Model, ResourceWithOwner):
    """A Model resource extended with ownership information for access control."""

    pass


class ShieldWithOwner(Shield, ResourceWithOwner):
    """A Shield resource extended with ownership information for access control."""

    pass


class VectorStoreWithOwner(VectorStore, ResourceWithOwner):
    """A VectorStore resource extended with ownership information for access control."""

    pass


class ToolGroupWithOwner(ToolGroup, ResourceWithOwner):
    """A ToolGroup resource extended with ownership information for access control."""

    pass


RoutableObject = Model | Shield | VectorStore | ToolGroup

RoutableObjectWithProvider = Annotated[
    ModelWithOwner | ShieldWithOwner | VectorStoreWithOwner | ToolGroupWithOwner,
    Field(discriminator="type"),
]

RoutedProtocol = Inference | Safety | VectorIO | ToolRuntime


# Example: /inference, /safety
class AutoRoutedProviderSpec(ProviderSpec):
    """Provider spec for automatically routed APIs like inference and safety that delegate to a routing table."""

    provider_type: str = "router"
    config_class: str = ""

    container_image: str | None = None
    routing_table_api: Api
    module: str
    provider_data_validator: str | None = Field(
        default=None,
    )


# Example: /models, /shields
class RoutingTableProviderSpec(ProviderSpec):
    """Provider spec for routing table APIs like models and shields that manage resource registries."""

    provider_type: str = "routing_table"
    config_class: str = ""
    container_image: str | None = None

    router_api: Api
    module: str
    pip_packages: list[str] = Field(default_factory=list)


class Provider(BaseModel):
    """A configured provider instance with its type, identifier, and configuration."""

    # provider_id of None means that the provider is not enabled - this happens
    # when the provider is enabled via a conditional environment variable
    provider_id: str | None
    provider_type: str
    config: dict[str, Any] = {}
    module: str | None = Field(
        default=None,
        description="""
 Fully-qualified name of the external provider module to import. The module is expected to have:

  - `get_adapter_impl(config, deps)`: returns the adapter implementation

  Example: `module: ramalama_stack`
 """,
    )


class BuildProvider(BaseModel):
    """A provider specification used during distribution build, containing just the type and optional module."""

    provider_type: str
    module: str | None = Field(
        default=None,
        description="""
 Fully-qualified name of the external provider module to import. The module is expected to have:

  - `get_adapter_impl(config, deps)`: returns the adapter implementation

  Example: `module: ramalama_stack`
 """,
    )


class DistributionSpec(BaseModel):
    """Specification defining a distribution's providers and container image."""

    description: str | None = Field(
        default="",
        description="Description of the distribution",
    )
    container_image: str | None = None
    providers: dict[str, list[BuildProvider]] = Field(
        default_factory=dict,
        description="""
        Provider Types for each of the APIs provided by this distribution. If you
        select multiple providers, you should provide an appropriate 'routing_map'
        in the runtime configuration to help route to the correct provider.
        """,
    )


class OAuth2JWKSConfig(BaseModel):
    """Configuration for OAuth2 JSON Web Key Set (JWKS) key retrieval."""

    # The JWKS URI for collecting public keys
    uri: str
    token: str | None = Field(default=None, description="token to authorise access to jwks")
    key_recheck_period: int = Field(default=3600, description="The period to recheck the JWKS URI for key updates")


class OAuth2IntrospectionConfig(BaseModel):
    """Configuration for OAuth2 token introspection endpoint (RFC 7662)."""

    url: str
    client_id: str
    client_secret: str
    send_secret_in_body: bool = False


class AuthProviderType(StrEnum):
    """Supported authentication provider types."""

    OAUTH2_TOKEN = "oauth2_token"
    GITHUB_TOKEN = "github_token"
    CUSTOM = "custom"
    KUBERNETES = "kubernetes"
    UPSTREAM_HEADER = "upstream_header"


class OAuth2TokenAuthConfig(BaseModel):
    """Configuration for OAuth2 token authentication."""

    type: Literal[AuthProviderType.OAUTH2_TOKEN] = AuthProviderType.OAUTH2_TOKEN
    audience: str = Field(default="ogx")
    verify_tls: bool = Field(default=True)
    tls_cafile: Path | None = Field(default=None)
    issuer: str | None = Field(default=None, description="The OIDC issuer URL.")
    claims_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "sub": "roles",
            "username": "roles",
            "groups": "teams",
            "team": "teams",
            "project": "projects",
            "tenant": "namespaces",
            "namespace": "namespaces",
        },
    )
    jwks: OAuth2JWKSConfig | None = Field(default=None, description="JWKS configuration")
    introspection: OAuth2IntrospectionConfig | None = Field(
        default=None, description="OAuth2 introspection configuration"
    )

    @classmethod
    @field_validator("claims_mapping")
    def validate_claims_mapping(cls, v):
        for key, value in v.items():
            if not value:
                raise ValueError(f"claims_mapping value cannot be empty: {key}")
        return v

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if not self.jwks and not self.introspection:
            raise ValueError("One of jwks or introspection must be configured")
        if self.jwks and self.introspection:
            raise ValueError("At present only one of jwks or introspection should be configured")
        return self


class CustomAuthConfig(BaseModel):
    """Configuration for custom authentication."""

    type: Literal[AuthProviderType.CUSTOM] = AuthProviderType.CUSTOM
    endpoint: str = Field(
        ...,
        description="Custom authentication endpoint URL",
    )


class GitHubTokenAuthConfig(BaseModel):
    """Configuration for GitHub token authentication."""

    type: Literal[AuthProviderType.GITHUB_TOKEN] = AuthProviderType.GITHUB_TOKEN
    github_api_base_url: str = Field(
        default="https://api.github.com",
        description="Base URL for GitHub API (use https://api.github.com for public GitHub)",
    )
    claims_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "login": "roles",
            "organizations": "teams",
        },
        description="Mapping from GitHub user fields to access attributes",
    )


class KubernetesAuthProviderConfig(BaseModel):
    """Configuration for Kubernetes authentication provider."""

    type: Literal[AuthProviderType.KUBERNETES] = AuthProviderType.KUBERNETES
    api_server_url: str = Field(
        default="https://kubernetes.default.svc",
        description="Kubernetes API server URL (e.g., https://api.cluster.domain:6443)",
    )
    verify_tls: bool = Field(default=True, description="Whether to verify TLS certificates")
    tls_cafile: Path | None = Field(default=None, description="Path to CA certificate file for TLS verification")
    claims_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "username": "roles",
            "groups": "roles",
        },
        description="Mapping of Kubernetes user claims to access attributes",
    )

    @field_validator("api_server_url")
    @classmethod
    def validate_api_server_url(cls, v):
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"api_server_url must be a valid URL with scheme and host: {v}")
        if parsed.scheme not in ["http", "https"]:
            raise ValueError(f"api_server_url scheme must be http or https: {v}")
        return v

    @field_validator("claims_mapping")
    @classmethod
    def validate_claims_mapping(cls, v):
        for key, value in v.items():
            if not value:
                raise ValueError(f"claims_mapping value cannot be empty: {key}")
        return v


class UpstreamHeaderAuthConfig(BaseModel):
    """Configuration for upstream header authentication.

    Used when an upstream gateway (Authorino, Istio, or any reverse proxy) handles
    authentication and injects user identity into request headers. OGX trusts
    these headers and extracts the principal and optional attributes from them.
    """

    type: Literal[AuthProviderType.UPSTREAM_HEADER] = AuthProviderType.UPSTREAM_HEADER
    principal_header: str = Field(
        description="HTTP header containing the authenticated user's identity (e.g. x-auth-user-id)",
    )
    attributes_header: str | None = Field(
        default=None,
        description="HTTP header containing JSON-encoded user attributes for access control (e.g. x-auth-attributes)",
    )


AuthProviderConfig = Annotated[
    OAuth2TokenAuthConfig
    | GitHubTokenAuthConfig
    | CustomAuthConfig
    | KubernetesAuthProviderConfig
    | UpstreamHeaderAuthConfig,
    Field(discriminator="type"),
]


class AuthenticationConfig(BaseModel):
    """Top-level authentication configuration."""

    provider_config: AuthProviderConfig | None = Field(
        default=None,
        description="Authentication provider configuration (optional if only using route_policy)",
    )
    route_policy: list[RouteAccessRule] = Field(
        default=[],
        description="Rules for determining access to API routes (infrastructure-level)",
    )
    access_policy: list[AccessRule] = Field(
        default=[],
        description="Rules for determining access to resources (data-level)",
    )


class AuthenticationRequiredError(Exception):
    """Raised when a request requires authentication but none was provided."""

    pass


class QualifiedModel(BaseModel):
    """A qualified model identifier, consisting of a provider ID and a model ID."""

    provider_id: str
    model_id: str
    embedding_dimensions: int | None = None


class RerankerModel(BaseModel):
    """A model identifier of a reranker model, consisting of a provider ID and a model ID."""

    provider_id: str
    model_id: str


class RewriteQueryParams(BaseModel):
    """Parameters for query rewriting/expansion."""

    model: QualifiedModel | None = Field(
        default=None,
        description="LLM model for query rewriting/expansion in vector search.",
    )
    prompt: str = Field(
        default="Expand this query with relevant synonyms and related terms. Return only the improved query, no explanations:\n\n{query}\n\nImproved query:",
        description="Prompt template for query rewriting. Use {query} as placeholder for the original query.",
    )
    max_tokens: int = Field(
        default=100,
        description="Maximum number of tokens for query expansion responses.",
    )
    temperature: float = Field(
        default=0.3,
        description="Temperature for query expansion model (0.0 = deterministic, 1.0 = creative).",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if "{query}" not in v:
            raise ValueError("prompt must contain {query} placeholder")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        if v > 4096:
            raise ValueError("max_tokens should not exceed 4096")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


class FileSearchParams(BaseModel):
    """Configuration for file search tool output formatting."""

    header_template: str = Field(
        default="file_search tool found {num_chunks} chunks:\nBEGIN of file_search tool results.\n",
        description="Template for the header text shown before search results. Available placeholders: {num_chunks} number of chunks found.",
    )
    footer_template: str = Field(
        default="END of file_search tool results.\n",
        description="Template for the footer text shown after search results.",
    )

    @field_validator("header_template")
    @classmethod
    def validate_header_template(cls, v: str) -> str:
        if len(v) == 0:
            raise ValueError("header_template must not be empty")
        if "{num_chunks}" not in v:
            raise ValueError("header_template must contain {num_chunks} placeholder")
        if "file_search" not in v.lower():
            raise ValueError("header_template must contain 'file_search' keyword to ensure proper tool identification")
        return v


class ContextPromptParams(BaseModel):
    """Configuration for LLM prompt content and chunk formatting."""

    chunk_annotation_template: str = Field(
        default="Result {index}\nContent: {chunk.content}\nMetadata: {metadata}\n",
        description="Template for formatting individual chunks in search results. Available placeholders: {index} 1-based chunk index, {chunk.content} chunk content, {metadata} chunk metadata dict.",
    )
    context_template: str = Field(
        default='The above results were retrieved to help answer the user\'s query: "{query}". Use them as supporting information only in answering this query. {annotation_instruction}\n',
        description="Template for explaining the search results to the model. Available placeholders: {query} user's query, {num_chunks} number of chunks.",
    )

    @field_validator("chunk_annotation_template")
    @classmethod
    def validate_chunk_annotation_template(cls, v: str) -> str:
        if len(v) == 0:
            raise ValueError("chunk_annotation_template must not be empty")
        if "{chunk.content}" not in v:
            raise ValueError("chunk_annotation_template must contain {chunk.content} placeholder")
        if "{index}" not in v:
            raise ValueError("chunk_annotation_template must contain {index} placeholder")
        return v

    @field_validator("context_template")
    @classmethod
    def validate_context_template(cls, v: str) -> str:
        if len(v) == 0:
            raise ValueError("context_template must not be empty")
        if "{query}" not in v:
            raise ValueError("context_template must contain {query} placeholder")
        return v


class AnnotationPromptParams(BaseModel):
    """Configuration for source annotation and attribution features."""

    enable_annotations: bool = Field(
        default=True,
        description="Whether to include annotation information in results.",
    )
    annotation_instruction_template: str = Field(
        default="Cite sources immediately at the end of sentences before punctuation, using `<|file-id|>` format like 'This is a fact <|file-Cn3MSNn72ENTiiq11Qda4A|>.'. Do not add extra punctuation. Use only the file IDs provided, do not invent new ones.",
        description="Instructions for how the model should cite sources. Used when enable_annotations is True.",
    )
    chunk_annotation_template: str = Field(
        default="[{index}] {metadata_text} cite as <|{file_id}|>\n{chunk_text}\n",
        description="Template for chunks with annotation information. Available placeholders: {index} 1-based chunk index, {metadata_text} formatted metadata, {file_id} document identifier, {chunk_text} chunk content.",
    )

    @field_validator("chunk_annotation_template")
    @classmethod
    def validate_chunk_annotation_template(cls, v: str) -> str:
        if len(v) == 0:
            raise ValueError("chunk_annotation_template must not be empty")
        if "{index}" not in v:
            raise ValueError("chunk_annotation_template must contain {index} placeholder")
        if "{chunk_text}" not in v:
            raise ValueError("chunk_annotation_template must contain {chunk_text} placeholder")
        if "{file_id}" not in v:
            raise ValueError("chunk_annotation_template must contain {file_id} placeholder")
        return v

    @field_validator("annotation_instruction_template")
    @classmethod
    def validate_annotation_instruction_template(cls, v: str) -> str:
        if len(v) == 0:
            raise ValueError("annotation_instruction_template must not be empty")
        return v


class FileIngestionParams(BaseModel):
    """Configuration for file processing during ingestion."""

    default_chunk_size_tokens: int = Field(
        default=512,
        description="Default chunk size for RAG tool operations when not specified",
    )
    default_chunk_overlap_tokens: int = Field(
        default=128,
        description="Default overlap in tokens between chunks (original default: 512 // 4 = 128)",
    )


class ChunkRetrievalParams(BaseModel):
    """Configuration for chunk retrieval and ranking during search."""

    chunk_multiplier: int = Field(
        default=5,
        description="Multiplier for OpenAI API over-retrieval (affects all providers)",
    )
    max_tokens_in_context: int = Field(
        default=4000,
        description="Maximum tokens allowed in RAG context before truncation",
    )
    default_reranker_strategy: str = Field(
        default="rrf",
        description="Default reranker when not specified: 'rrf', 'weighted', or 'normalized'",
    )
    rrf_impact_factor: float = Field(
        default=60.0,
        description="Impact factor for RRF (Reciprocal Rank Fusion) reranking",
    )
    weighted_search_alpha: float = Field(
        default=0.5,
        description="Alpha weight for weighted search reranking (0.0-1.0)",
    )
    default_search_mode: str = Field(
        default="vector",
        description="Default search mode: 'vector', 'keyword', or 'hybrid'",
    )


class FileBatchParams(BaseModel):
    """Configuration for file batch processing."""

    max_concurrent_files_per_batch: int = Field(
        default=3,
        description="Maximum files processed concurrently in file batches",
    )
    file_batch_chunk_size: int = Field(
        default=10,
        description="Number of files to process in each batch chunk",
    )
    cleanup_interval_seconds: int = Field(
        default=86400,  # 24 hours
        description="Interval for cleaning up expired file batches (seconds)",
    )


class ContextualRetrievalParams(BaseModel):
    """Configuration for contextual retrieval during file ingestion.

    Contextual retrieval uses an LLM to generate situational context for each chunk
    before embedding, improving search quality. See Anthropic's Contextual Retrieval.
    """

    model: QualifiedModel | None = Field(
        default=None,
        description="Default LLM model for contextual retrieval. Used when model_id is not specified in chunking strategy.",
    )
    default_timeout_seconds: int = Field(
        default=120,
        ge=10,
        description="Default timeout in seconds for each LLM contextualization call.",
    )
    default_max_concurrency: int = Field(
        default=3,
        ge=1,
        description="Default maximum concurrent LLM calls for contextualization.",
    )
    max_document_tokens: int = Field(
        default=100000,
        ge=1000,
        description="Maximum document size in tokens. Documents exceeding this are rejected for contextual retrieval.",
    )


class VectorStoresConfig(BaseModel):
    """Configuration for vector stores in the stack."""

    default_provider_id: str | None = Field(
        default=None,
        description="ID of the vector_io provider to use as default when multiple providers are available and none is specified.",
    )
    default_embedding_model: QualifiedModel | None = Field(
        default=None,
        description="Default embedding model configuration for vector stores.",
    )
    default_reranker_model: RerankerModel | None = Field(
        default=None,
        description="Default reranker model configuration for vector stores.",
    )
    rewrite_query_params: RewriteQueryParams | None = Field(
        default=None,
        description="Parameters for query rewriting/expansion. None disables query rewriting.",
    )
    file_search_params: FileSearchParams = Field(
        default_factory=FileSearchParams,
        description="Configuration for file search tool output formatting.",
    )
    context_prompt_params: ContextPromptParams = Field(
        default_factory=ContextPromptParams,
        description="Configuration for LLM prompt content and chunk formatting.",
    )
    annotation_prompt_params: AnnotationPromptParams = Field(
        default_factory=AnnotationPromptParams,
        description="Configuration for source annotation and attribution features.",
    )

    file_ingestion_params: FileIngestionParams = Field(
        default_factory=FileIngestionParams,
        description="Configuration for file processing during ingestion.",
    )
    chunk_retrieval_params: ChunkRetrievalParams = Field(
        default_factory=ChunkRetrievalParams,
        description="Configuration for chunk retrieval and ranking during search.",
    )
    file_batch_params: FileBatchParams = Field(
        default_factory=FileBatchParams,
        description="Configuration for file batch processing.",
    )
    contextual_retrieval_params: ContextualRetrievalParams = Field(
        default_factory=ContextualRetrievalParams,
        description="Configuration for contextual retrieval during file ingestion.",
    )


class SafetyConfig(BaseModel):
    """Configuration for default moderations model."""

    default_shield_id: str | None = Field(
        default=None,
        description="ID of the shield to use for when `model` is not specified in the `moderations` API request.",
    )


class QuotaPeriod(StrEnum):
    """Time period for request quota enforcement."""

    DAY = "day"


class QuotaConfig(BaseModel):
    """Configuration for per-client request rate limiting."""

    kvstore: KVStoreReference = Field(description="Config for KV store backend (SQLite only for now)")
    anonymous_max_requests: int = Field(default=100, description="Max requests for unauthenticated clients per period")
    authenticated_max_requests: int = Field(
        default=1000, description="Max requests for authenticated clients per period"
    )
    period: QuotaPeriod = Field(default=QuotaPeriod.DAY, description="Quota period to set")


class CORSConfig(BaseModel):
    """Configuration for Cross-Origin Resource Sharing (CORS) headers."""

    allow_origins: list[str] = Field(default_factory=list)
    allow_origin_regex: str | None = Field(default=None)
    allow_methods: list[str] = Field(default=["OPTIONS"])
    allow_headers: list[str] = Field(default_factory=list)
    allow_credentials: bool = Field(default=False)
    expose_headers: list[str] = Field(default_factory=list)
    max_age: int = Field(default=600, ge=0)

    @model_validator(mode="after")
    def validate_credentials_config(self) -> Self:
        if self.allow_credentials and (self.allow_origins == ["*"] or "*" in self.allow_origins):
            raise ValueError("Cannot use wildcard origins with credentials enabled")
        return self


def process_cors_config(cors_config: bool | CORSConfig | None) -> CORSConfig | None:
    """Convert a CORS configuration value into a resolved CORSConfig object.

    Args:
        cors_config: A boolean (True for dev defaults, False/None to disable), or a CORSConfig instance.

    Returns:
        A CORSConfig instance or None if CORS is disabled.
    """
    if cors_config is False or cors_config is None:
        return None

    if cors_config is True:
        # dev mode: allow localhost on any port
        return CORSConfig(
            allow_origins=[],
            allow_origin_regex=r"https?://localhost:\d+",
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
        )

    if isinstance(cors_config, CORSConfig):
        return cors_config

    raise ValueError(f"Expected bool or CORSConfig, got {type(cors_config).__name__}")


class RegisteredResources(BaseModel):
    """Registry of resources available in the distribution."""

    models: list[ModelInput] = Field(default_factory=list)
    shields: list[ShieldInput] = Field(default_factory=list)
    vector_stores: list[VectorStoreInput] = Field(default_factory=list)
    tool_groups: list[ToolGroupInput] = Field(default_factory=list, deprecated=True)

    @model_validator(mode="after")
    def _warn_deprecated_tool_groups(self) -> Self:
        if self.tool_groups:
            warnings.warn(
                "'registered_resources.tool_groups' is deprecated and will be removed in a future release. "
                "Built-in tool groups are now auto-registered based on configured tool_runtime providers. "
                "Please remove 'tool_groups' from your configuration.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.tool_groups = []
        return self


class ServerConfig(BaseModel):
    """Configuration for the HTTP(S) server including TLS, authentication, and quotas."""

    port: int = Field(
        default=8321,
        description="Port to listen on",
        ge=1024,
        le=65535,
    )
    tls_certfile: str | None = Field(
        default=None,
        description="Path to TLS certificate file for HTTPS",
    )
    tls_keyfile: str | None = Field(
        default=None,
        description="Path to TLS key file for HTTPS",
    )
    tls_cafile: str | None = Field(
        default=None,
        description="Path to TLS CA file for HTTPS with mutual TLS authentication",
    )
    auth: AuthenticationConfig | None = Field(
        default=None,
        description="Authentication configuration for the server",
    )
    host: str | None = Field(
        default=None,
        description="The host the server should listen on",
    )
    quota: QuotaConfig | None = Field(
        default=None,
        description="Per client quota request configuration",
    )
    cors: bool | CORSConfig | None = Field(
        default=None,
        description="CORS configuration for cross-origin requests. Can be:\n"
        "- true: Enable localhost CORS for development\n"
        "- {allow_origins: [...], allow_methods: [...], ...}: Full configuration",
    )
    workers: int = Field(
        default=1,
        description="Number of workers to use for the server",
    )
    registry_refresh_interval_seconds: int = Field(
        default=300,
        description="Interval in seconds between registry refreshes for syncing model information from providers",
        gt=0,
    )


class StackConfig(BaseModel):
    """Top-level runtime configuration for a OGX distribution including providers, storage, and server settings."""

    version: int = OGX_RUN_CONFIG_VERSION

    distro_name: str | None = Field(
        default=None,
        description="""
Reference to the distribution this package refers to. For unregistered (adhoc) packages,
this could be just a hash
""",
    )
    image_name: str | None = Field(
        default=None,
        deprecated=True,
        description="""
DEPRECATED: Use 'distro_name' instead. This field is maintained for backward compatibility.
Reference to the distribution this package refers to. For unregistered (adhoc) packages,
this could be just a hash
""",
    )
    container_image: str | None = Field(
        default=None,
        description="Reference to the container image if this package refers to a container",
    )
    apis: list[str] = Field(
        default_factory=list,
        description="""
The list of APIs to serve. If not specified, all APIs specified in the provider_map will be served""",
    )

    providers: dict[str, list[Provider]] = Field(
        description="""
One or more providers to use for each API. The same provider_type (e.g., builtin)
can be instantiated multiple times (with different configs) if necessary.
""",
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Catalog of named storage backends and references available to the stack",
    )

    registered_resources: RegisteredResources = Field(
        default_factory=RegisteredResources,
        description="Registry of resources available in the distribution",
    )

    logging: LoggingConfig | None = Field(default=None, description="Configuration for OGX Logging")

    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="Configuration for the HTTP(S) server",
    )

    external_providers_dir: Path | None = Field(
        default=None,
        description="Path to directory containing external provider implementations. The providers code and dependencies must be installed on the system.",
    )

    external_apis_dir: Path | None = Field(
        default=None,
        description="Path to directory containing external API implementations. The APIs code and dependencies must be installed on the system.",
    )

    vector_stores: VectorStoresConfig | None = Field(
        default=None,
        description="Configuration for vector stores, including default embedding model",
    )

    safety: SafetyConfig | None = Field(
        default=None,
        description="Configuration for default moderations model",
    )

    connectors: list[ConnectorInput] = Field(
        default_factory=list,
        description="List of connectors to register at stack startup",
    )

    @field_validator("external_providers_dir")
    @classmethod
    def validate_external_providers_dir(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @model_validator(mode="after")
    def validate_distro_name_migration(self) -> "StackConfig":
        """Handle migration from image_name to distro_name."""
        import warnings

        if self.distro_name is None and self.image_name is None:
            raise ValueError("Either 'distro_name' or 'image_name' must be provided")

        if self.image_name is not None and self.distro_name is None:
            # Migrate from image_name to distro_name
            warnings.warn(
                "The 'image_name' field is deprecated. Please use 'distro_name' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.distro_name = self.image_name
        elif self.image_name is not None and self.distro_name is not None:
            # Both provided - warn and prefer distro_name
            warnings.warn(
                "Both 'image_name' and 'distro_name' were provided. "
                "The 'image_name' field is deprecated and will be ignored. "
                "Please use only 'distro_name' in your configuration.",
                DeprecationWarning,
                stacklevel=2,
            )

        return self

    @model_validator(mode="after")
    def validate_server_stores(self) -> "StackConfig":
        backend_map = self.storage.backends
        stores = self.storage.stores
        kv_backends = {
            name
            for name, cfg in backend_map.items()
            if cfg.type
            in {
                StorageBackendType.KV_REDIS,
                StorageBackendType.KV_SQLITE,
                StorageBackendType.KV_POSTGRES,
                StorageBackendType.KV_MONGODB,
            }
        }
        sql_backends = {
            name
            for name, cfg in backend_map.items()
            if cfg.type in {StorageBackendType.SQL_SQLITE, StorageBackendType.SQL_POSTGRES}
        }

        def _ensure_backend(reference, expected_set, store_name: str) -> None:
            if reference is None:
                return
            backend_name = reference.backend
            if backend_name not in backend_map:
                raise ValueError(
                    f"{store_name} references unknown backend '{backend_name}'. "
                    f"Available backends: {sorted(backend_map)}"
                )
            if backend_name not in expected_set:
                raise ValueError(
                    f"{store_name} references backend '{backend_name}' of type "
                    f"'{backend_map[backend_name].type.value}', but a backend of type "
                    f"{'kv_*' if expected_set is kv_backends else 'sql_*'} is required."
                )

        _ensure_backend(stores.metadata, kv_backends, "storage.stores.metadata")
        _ensure_backend(stores.inference, sql_backends, "storage.stores.inference")
        _ensure_backend(stores.conversations, sql_backends, "storage.stores.conversations")
        _ensure_backend(stores.responses, sql_backends, "storage.stores.responses")
        _ensure_backend(stores.prompts, kv_backends, "storage.stores.prompts")
        return self
