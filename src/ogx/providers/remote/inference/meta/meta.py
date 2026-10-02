# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.remote.inference.meta.config import MetaConfig
from ogx.providers.utils.inference.anthropic_mixin import AnthropicMixin
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin


class MetaInferenceAdapter(AnthropicMixin, OpenAIMixin):
    """Inference adapter for the Meta AI OpenAI-compatible API endpoint (api.meta.ai).

    Chat Completions and the Responses API are served by the OpenAI-compatible
    mixin. The endpoint also exposes the Anthropic Messages API natively, so
    ``anthropic_messages``/``anthropic_count_tokens`` forward directly to
    ``/v1/messages`` (via :class:`AnthropicMixin`) instead of using the mixin's
    translation fallback.
    """

    config: MetaConfig

    provider_data_api_key_field: str = "meta_api_key"

    def get_base_url(self) -> str:
        """Return the Meta AI API base URL (including the /v1 suffix)."""
        return str(self.config.base_url)
