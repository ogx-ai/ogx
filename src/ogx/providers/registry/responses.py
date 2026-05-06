# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.core.storage.kvstore import kvstore_dependencies
from ogx_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
)


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
                "scikit-learn",
                "mcp>=1.23.0",
            ]
            + kvstore_dependencies(),  # TODO make this dynamic based on the kvstore config
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
            optional_api_dependencies=[
                Api.safety,
            ],
            description="Meta's reference implementation of an agent system that can use tools, access vector databases, and perform complex reasoning tasks.",
        ),
    ]
