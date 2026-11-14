# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import Field, HttpUrl

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


@json_schema_type
class TextEmbeddingsInferenceConfig(RemoteInferenceProviderConfig):
    """Configuration for the HuggingFace Text-Embeddings-Inference provider."""

    base_url: HttpUrl | None = Field(
        default=HttpUrl("http://localhost:8080/v1"),
        description="Base URL for the Text-Embeddings-Inference server (OpenAI-compatible /v1 endpoint)",
    )

    @classmethod
    def sample_run_config(
        cls,
        base_url: str = "${env.TEI_URL:=http://localhost:8080/v1}",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "base_url": base_url,
        }
