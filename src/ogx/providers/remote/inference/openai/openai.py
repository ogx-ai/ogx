# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.log import get_logger
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import OpenAIConfig

logger = get_logger(name=__name__, category="inference::openai")


#
# This OpenAI adapter implements Inference methods using OpenAIMixin
#
class OpenAIInferenceAdapter(OpenAIMixin):
    """
    OpenAI Inference Adapter for OGX.
    """

    config: OpenAIConfig

    provider_data_api_key_field: str = "openai_api_key"

    supports_tokenized_embeddings_input: bool = True

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "text-embedding-3-small": {"embedding_dimension": 1536, "context_length": 8192},
        "text-embedding-3-large": {"embedding_dimension": 3072, "context_length": 8192},
    }

    async def openai_chat_completions_with_reasoning(self, params) -> None:
        raise ValueError(
            "OpenAI provider does not support reasoning. "
            "Please remove the reasoning parameter from your request or choose another provider."
        )

    def get_base_url(self) -> str:
        """
        Get the OpenAI API base URL.

        Returns the OpenAI API base URL from the configuration.
        """
        return str(self.config.base_url)
