# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Provider SDK surface for OGX.

This namespace contains the symbols an out-of-tree provider needs to register
itself with the OGX server: protocol classes, provider specs, the `Api` enum,
the `webmethod` decorator, schema utilities, and shared resource/value types.

The Provider SDK surface is levelled as a single cohesive contract — see
`docs/docs/concepts/apis/api_leveling.mdx` for the stability rules. The whole
surface is `v1`-stable; removals or renames require a major version bump of
the `ogx-api` package.

Symbols re-exported here also remain importable from the top level (`from
ogx_api import X`) for backwards compatibility. New code should prefer
`from ogx_api.provider import X` to make the contract explicit.
"""

# Protocol classes — the abstract interfaces a provider implements.
from ogx_api.admin import Admin
from ogx_api.batches import Batches
from ogx_api.connectors import Connectors
from ogx_api.conversations import Conversations
from ogx_api.datatypes import (
    Api,
    DynamicApiMeta,
    Error,
    ExternalApiSpec,
    HealthResponse,
    HealthStatus,
    InlineProviderSpec,
    ModelsProtocolPrivate,
    ProviderSpec,
    RemoteProviderConfig,
    RemoteProviderSpec,
    RoutingTable,
    ShieldsProtocolPrivate,
    ToolGroupsProtocolPrivate,
    VectorStoresProtocolPrivate,
)
from ogx_api.file_processors import FileProcessors
from ogx_api.files import Files
from ogx_api.inference import Inference, InferenceProvider, ModelStore
from ogx_api.inspect_api import Inspect
from ogx_api.interactions import Interactions
from ogx_api.messages import Messages

# Shared resource/value types. Canonical home is `ogx_api.types`; re-exported
# here because providers import them as part of registration / protocol
# implementations.
from ogx_api.models import Model, Models
from ogx_api.prompts import Prompts
from ogx_api.providers import Providers
from ogx_api.resource import Resource, ResourceType
from ogx_api.responses import Responses
from ogx_api.safety import Safety, ShieldStore
from ogx_api.schema_utils import (
    CallableT,
    ExtraBodyField,
    SchemaInfo,
    clear_dynamic_schema_types,
    get_registered_schema_info,
    iter_dynamic_schema_types,
    iter_json_schema_types,
    iter_registered_schema_types,
    json_schema_type,
    register_dynamic_schema_type,
    register_schema,
)
from ogx_api.shields import Shield, Shields
from ogx_api.tools import ToolGroup, ToolGroups, ToolRuntime, ToolStore
from ogx_api.validators import validate_embeddings_input_is_text
from ogx_api.vector_io import VectorIO
from ogx_api.vector_stores import VectorStore
from ogx_api.version import OGX_API_V1, OGX_API_V1ALPHA, OGX_API_V1BETA

__all__ = [
    # Core provider machinery
    "Api",
    "DynamicApiMeta",
    "Error",
    "ExternalApiSpec",
    "HealthResponse",
    "HealthStatus",
    "InlineProviderSpec",
    "ProviderSpec",
    "RemoteProviderConfig",
    "RemoteProviderSpec",
    "RoutingTable",
    # Protocol-private mixins
    "ModelsProtocolPrivate",
    "ShieldsProtocolPrivate",
    "ToolGroupsProtocolPrivate",
    "VectorStoresProtocolPrivate",
    # Resource base classes
    "Resource",
    "ResourceType",
    # Schema utilities
    "CallableT",
    "ExtraBodyField",
    "SchemaInfo",
    "clear_dynamic_schema_types",
    "get_registered_schema_info",
    "iter_dynamic_schema_types",
    "iter_json_schema_types",
    "iter_registered_schema_types",
    "json_schema_type",
    "register_dynamic_schema_type",
    "register_schema",
    # Version constants
    "OGX_API_V1",
    "OGX_API_V1ALPHA",
    "OGX_API_V1BETA",
    # Validators
    "validate_embeddings_input_is_text",
    # API protocol classes
    "Admin",
    "Batches",
    "Connectors",
    "Conversations",
    "FileProcessors",
    "Files",
    "Inference",
    "InferenceProvider",
    "ModelStore",
    "Inspect",
    "Interactions",
    "Messages",
    "Models",
    "Prompts",
    "Providers",
    "Responses",
    "Safety",
    "ShieldStore",
    "Shields",
    "ToolGroups",
    "ToolRuntime",
    "ToolStore",
    "VectorIO",
    # Shared resource/value types (canonical home in ogx_api.types)
    "Model",
    "Shield",
    "ToolGroup",
    "VectorStore",
]
