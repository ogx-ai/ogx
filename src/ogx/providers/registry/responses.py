# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
)

# All possible kvstore dependencies for registry/provider specifications.
# NOTE: For specific kvstore implementations, use config.pip_packages instead.
# This is the union of all dependencies for cases where the specific kvstore type
# is not known at declaration time (e.g., provider registries).
KVSTORE_DEPS = ["aiosqlite", "asyncpg", "redis", "pymongo>=4.18.1"]  # CVE-2026-88029: query-operator injection


def available_providers() -> list[ProviderSpec]:
    """Return the list of available agent provider specifications.

    Returns:
        List of ProviderSpec objects describing available providers
    """
    return [
        InlineProviderSpec(
            api=Api.responses,
            provider_type="inline::builtin",
            pip_packages=[
                "matplotlib",
                "fonttools>=4.60.2",
                "pillow",
                "pandas",
                "mcp>=1.28.1,<2.0",
            ]
            + KVSTORE_DEPS,  # TODO make this dynamic based on the kvstore config
            module="ogx.providers.inline.responses.builtin",
            config_class="ogx.providers.inline.responses.builtin.BuiltinResponsesImplConfig",
            api_dependencies=[
                Api.inference,
                Api.vector_io,
                Api.tool_runtime,
                Api.tool_groups,
                Api.conversations,
                Api.prompts,
                Api.files,
                Api.connectors,
            ],
            optional_api_dependencies=[Api.skills],
            description="Meta's reference implementation of an agent system that can use tools, access vector databases, and perform complex reasoning tasks.",
        ),
    ]
