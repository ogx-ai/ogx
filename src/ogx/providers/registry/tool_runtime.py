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
                "scikit-learn",
                "scipy",
                "nltk>=3.9.4",
                "sentencepiece",
                "transformers",
            ],
            module="ogx.providers.inline.tool_runtime.file_search",
            config_class="ogx.providers.inline.tool_runtime.file_search.config.FileSearchToolRuntimeConfig",
            api_dependencies=[Api.vector_io, Api.inference, Api.files],
            toolgroup_id="builtin::file_search",
            description="File search tool runtime for document ingestion, chunking, and semantic search.",
        ),
        RemoteProviderSpec(
            api=Api.tool_runtime,
            adapter_type="brave-search",
            provider_type="remote::brave-search",
            module="ogx.providers.remote.tool_runtime.brave_search",
            config_class="ogx.providers.remote.tool_runtime.brave_search.config.BraveSearchToolConfig",
            pip_packages=["requests"],
            provider_data_validator="ogx.providers.remote.tool_runtime.brave_search.BraveSearchToolProviderDataValidator",
            toolgroup_id="builtin::websearch",
            description="Brave Search tool for web search capabilities with privacy-focused results.",
        ),
        RemoteProviderSpec(
            api=Api.tool_runtime,
            adapter_type="bing-search",
            provider_type="remote::bing-search",
            module="ogx.providers.remote.tool_runtime.bing_search",
            config_class="ogx.providers.remote.tool_runtime.bing_search.config.BingSearchToolConfig",
            pip_packages=["requests"],
            provider_data_validator="ogx.providers.remote.tool_runtime.bing_search.BingSearchToolProviderDataValidator",
            toolgroup_id="builtin::websearch",
            description="Bing Search tool for web search capabilities using Microsoft's search engine.",
        ),
        RemoteProviderSpec(
            api=Api.tool_runtime,
            adapter_type="tavily-search",
            provider_type="remote::tavily-search",
            module="ogx.providers.remote.tool_runtime.tavily_search",
            config_class="ogx.providers.remote.tool_runtime.tavily_search.config.TavilySearchToolConfig",
            pip_packages=["requests"],
            provider_data_validator="ogx.providers.remote.tool_runtime.tavily_search.TavilySearchToolProviderDataValidator",
            toolgroup_id="builtin::websearch",
            description="Tavily Search tool for AI-optimized web search with structured results.",
        ),
        RemoteProviderSpec(
            api=Api.tool_runtime,
            adapter_type="wolfram-alpha",
            provider_type="remote::wolfram-alpha",
            module="ogx.providers.remote.tool_runtime.wolfram_alpha",
            config_class="ogx.providers.remote.tool_runtime.wolfram_alpha.config.WolframAlphaToolConfig",
            pip_packages=["requests"],
            provider_data_validator="ogx.providers.remote.tool_runtime.wolfram_alpha.WolframAlphaToolProviderDataValidator",
            toolgroup_id="builtin::wolfram_alpha",
            description="Wolfram Alpha tool for computational knowledge and mathematical calculations.",
        ),
        RemoteProviderSpec(
            api=Api.tool_runtime,
            adapter_type="model-context-protocol",
            provider_type="remote::model-context-protocol",
            module="ogx.providers.remote.tool_runtime.model_context_protocol",
            config_class="ogx.providers.remote.tool_runtime.model_context_protocol.config.MCPProviderConfig",
            pip_packages=["mcp>=1.23.0"],
            provider_data_validator="ogx.providers.remote.tool_runtime.model_context_protocol.config.MCPProviderDataValidator",
            description="Model Context Protocol (MCP) tool for standardized tool calling and context management.",
        ),
    ]
