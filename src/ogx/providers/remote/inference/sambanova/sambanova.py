# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import SambaNovaImplConfig


class SambaNovaInferenceAdapter(OpenAIMixin):
    """Inference adapter for SambaNova AI platform."""

    config: SambaNovaImplConfig

    provider_data_api_key_field: str = "sambanova_api_key"
    download_images: bool = True  # SambaNova does not support image downloads server-size, perform them on the client
    """
    SambaNova Inference Adapter for OGX.
    """

    def get_base_url(self) -> str:
        """
        Get the base URL for OpenAI mixin.

        :return: The SambaNova base URL
        """
        return str(self.config.base_url)
