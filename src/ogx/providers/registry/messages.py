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
                "Implements the Anthropic Messages API by delegating to the inference API's "
                "anthropic_messages() method. OpenAIMixin provides default translation via "
                "openai_chat_completion. Providers with native /v1/messages support "
                "(e.g., Ollama, vLLM) override with direct passthrough. Message batch "
                "operations are implemented locally."
            ),
        ),
    ]
