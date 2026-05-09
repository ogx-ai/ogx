# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import Field, HttpUrl, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


class OllamaImplConfig(RemoteInferenceProviderConfig):
    """Configuration for the Ollama inference provider."""

    auth_credential: SecretStr | None = Field(default=None, exclude=True)

    base_url: HttpUrl | None = Field(default=HttpUrl(DEFAULT_OLLAMA_URL))

    @classmethod
    def sample_run_config(
        cls, base_url: str = "${env.OLLAMA_URL:=http://localhost:11434/v1}", **kwargs
    ) -> dict[str, Any]:
        return {
            "base_url": base_url,
        }
