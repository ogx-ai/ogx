# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import AzureConfig


class AzureInferenceAdapter(OpenAIMixin):
    """Inference adapter for Azure OpenAI Service."""

    config: AzureConfig

    provider_data_api_key_field: str = "azure_api_key"

    def get_base_url(self) -> str:
        """
        Get the Azure API base URL.

        Returns the Azure API base URL from the configuration.
        """
        return str(self.config.base_url)
