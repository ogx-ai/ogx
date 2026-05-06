# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


class NVIDIAProviderDataValidator(BaseModel):
    """Validates provider-specific request data for NVIDIA inference."""

    nvidia_api_key: SecretStr | None = Field(
        default=None,
        description="API key for NVIDIA NIM models",
    )


@json_schema_type
class NVIDIAConfig(RemoteInferenceProviderConfig):
    """
    Configuration for the NVIDIA NIM inference endpoint.

    Attributes:
        url (str): A base url for accessing the NVIDIA NIM, e.g. http://localhost:8000
        api_key (str): The access key for the hosted NIM endpoints
        rerank_model_to_url (dict[str, str]): Mapping of rerank model identifiers to their API endpoints

    There are two ways to access NVIDIA NIMs -
     0. Hosted: Preview APIs hosted at https://integrate.api.nvidia.com
     1. Self-hosted: You can run NVIDIA NIMs on your own infrastructure

    By default the configuration is set to use the hosted APIs. This requires
    an API key which can be obtained from https://ngc.nvidia.com/.

    By default the configuration will attempt to read the NVIDIA_API_KEY environment
    variable to set the api_key. Please do not put your API key in code.

    If you are using a self-hosted NVIDIA NIM, you can set the url to the
    URL of your running NVIDIA NIM and do not need to set the api_key.
    """

    base_url: HttpUrl | None = Field(
        default_factory=lambda: os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        description="A base url for accessing the NVIDIA NIM",
    )
    timeout: int = Field(
        default=60,
        description="Timeout for the HTTP requests",
    )
    rerank_model_to_url: dict[str, str] = Field(
        default_factory=lambda: {
            "nv-rerank-qa-mistral-4b:1": "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
            "nvidia/nv-rerankqa-mistral-4b-v3": "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking",
            "nvidia/llama-3.2-nv-rerankqa-1b-v2": "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
        },
        description="Mapping of rerank model identifiers to their API endpoints. ",
    )

    @classmethod
    def sample_run_config(
        cls,
        base_url: HttpUrl | None = "${env.NVIDIA_BASE_URL:=https://integrate.api.nvidia.com/v1}",
        api_key: str = "${env.NVIDIA_API_KEY:=}",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "base_url": base_url,
            "api_key": api_key,
        }
