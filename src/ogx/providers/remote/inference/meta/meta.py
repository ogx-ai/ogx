# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.log import get_logger
from ogx.providers.remote.inference.meta.config import MetaConfig
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin

logger = get_logger(name=__name__, category="inference::meta")


class MetaInferenceAdapter(OpenAIMixin):
    """Inference adapter for the Meta AI OpenAI-compatible API endpoint (api.meta.ai).

    The endpoint natively exposes the Chat Completions, Responses, and Anthropic
    Messages APIs, so this adapter relies entirely on the OpenAI-compatible mixin
    for chat completions and lets the built-in responses and messages providers
    layer on top of it.
    """

    config: MetaConfig

    provider_data_api_key_field: str = "meta_api_key"

    def get_base_url(self) -> str:
        """Return the Meta AI API base URL."""
        return str(self.config.base_url)
