# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import NovitaImplConfig


class NovitaInferenceAdapter(OpenAIMixin):
    """Inference adapter for the Novita AI platform.

    Novita exposes an OpenAI-compatible chat completions API, so the shared
    `OpenAIMixin` handles requests once pointed at Novita's base URL. See
    https://novita.ai/docs/guides/llm-api.
    """

    config: NovitaImplConfig

    provider_data_api_key_field: str = "novita_api_key"

    def get_base_url(self) -> str:
        return str(self.config.base_url)
