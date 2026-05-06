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


def available_providers() -> list[ProviderSpec]:
    """Return the list of available messages provider specifications."""
    return [
        InlineProviderSpec(
            api=Api.messages,
            provider_type="inline::builtin",
            pip_packages=[],
            module="ogx.providers.inline.messages",
            config_class="ogx.providers.inline.messages.config.MessagesConfig",
            api_dependencies=[
                Api.inference,
            ],
            description=(
                "Implements the Anthropic Messages API with two modes: native passthrough for providers "
                "that support /v1/messages natively (e.g. Ollama, vLLM), and automatic translation for "
                "all other providers by converting between Anthropic and OpenAI Chat Completions formats."
            ),
        ),
    ]
