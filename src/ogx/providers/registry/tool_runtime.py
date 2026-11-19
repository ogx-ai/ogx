# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.providers.registry.vector_io import DEFAULT_VECTOR_IO_DEPS
from ogx_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
    RemoteProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    """Return the list of available tool runtime provider specifications.

    Returns:
        List of ProviderSpec objects describing available providers
    """
    return [
        InlineProviderSpec(
            api=Api.tool_runtime,
            provider_type="inline::file-search",
            pip_packages=DEFAULT_VECTOR_IO_DEPS
            + [
                "tqdm",
                "numpy",
                "scipy",
                "sentencepiece",
                "transformers",
            ],
            module="ogx.providers.inline.tool_runtime.file_search",
            config_class="ogx.providers.inline.tool_runtime.file_search.config.FileSearchToolRuntimeConfig",
            api_dependencies=[Api.vector_io, Api.inference],
            toolgroup_id="builtin::file_search",
            description="File search tool runtime for document ingestion, chunking, and semantic search.",
        ),
        RemoteProviderSpec(
            api=Api.tool_runtime,
            adapter_type="model-context-protocol",
            provider_type="remote::model-context-protocol",
            module="ogx.providers.remote.tool_runtime.model_context_protocol",
            config_class="ogx.providers.remote.tool_runtime.model_context_protocol.config.MCPProviderConfig",
            pip_packages=["mcp>=2.0"],
            provider_data_validator="ogx.providers.remote.tool_runtime.model_context_protocol.config.MCPProviderDataValidator",
            description="Model Context Protocol (MCP) tool for standardized tool calling and context management.",
        ),
    ]
