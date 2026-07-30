# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.providers.remote.inference.groq.config import GroqConfig
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin


class GroqInferenceAdapter(OpenAIMixin):
    """Inference adapter for the Groq LPU platform."""

    config: GroqConfig

    provider_data_api_key_field: str = "groq_api_key"

    def get_base_url(self) -> str:
        return str(self.config.base_url)
