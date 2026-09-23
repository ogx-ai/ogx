# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.providers.remote.inference.llama_cpp_server.config import LlamaCppServerConfig
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import Model, ModelType


class LlamaCppServerInferenceAdapter(OpenAIMixin):
    """Inference adapter for llama.cpp servers with an OpenAI-compatible API."""

    config: LlamaCppServerConfig

    def get_api_key(self) -> str | None:
        if self.config.auth_credential is None:
            return "NO KEY REQUIRED"
        return self.config.auth_credential.get_secret_value()

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    def construct_model_from_identifier(self, identifier: str) -> Model:
        # llama.cpp's /v1/models response does not expose a model task/type field
        # (a router can serve llm and embedding models on one endpoint), so classify
        # by name, mirroring the vLLM adapter.
        lowered = identifier.lower()
        if "embed" in lowered:
            return Model(
                provider_id=self.__provider_id__,
                provider_resource_id=identifier,
                identifier=identifier,
                model_type=ModelType.embedding,
            )
        return super().construct_model_from_identifier(identifier)
