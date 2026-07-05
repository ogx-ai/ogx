# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import MistralImplConfig


class MistralInferenceAdapter(OpenAIMixin):
    """Inference adapter for the Mistral AI platform.

    Mistral exposes an OpenAI-compatible API (chat completions and embeddings),
    so the shared `OpenAIMixin` handles requests once pointed at Mistral's base
    URL. See https://docs.mistral.ai/api/.
    """

    config: MistralImplConfig

    provider_data_api_key_field: str = "mistral_api_key"

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "mistral-embed": {"embedding_dimension": 1024, "context_length": 8192},
    }

    def get_base_url(self) -> str:
        return str(self.config.base_url)
