# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.log import get_logger
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import FireworksImplConfig

logger = get_logger(name=__name__, category="inference::fireworks")


class FireworksInferenceAdapter(OpenAIMixin):
    """Inference adapter for the Fireworks AI platform."""

    config: FireworksImplConfig

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "nomic-ai/nomic-embed-text-v1.5": {"embedding_dimension": 768, "context_length": 8192},
        "accounts/fireworks/models/qwen3-embedding-8b": {"embedding_dimension": 4096, "context_length": 40960},
    }

    provider_data_api_key_field: str = "fireworks_api_key"

    def get_base_url(self) -> str:
        return str(self.config.base_url)
