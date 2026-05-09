# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator, Iterable

from databricks.sdk import WorkspaceClient

from ogx.log import get_logger
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import OpenAICompletion, OpenAICompletionRequestWithExtraBody

from .config import DatabricksImplConfig

logger = get_logger(name=__name__, category="inference::databricks")


class DatabricksInferenceAdapter(OpenAIMixin):
    """Inference adapter for the Databricks platform."""

    config: DatabricksImplConfig

    provider_data_api_key_field: str = "databricks_api_token"

    # source: https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models
    embedding_model_metadata: dict[str, dict[str, int]] = {
        "databricks-gte-large-en": {"embedding_dimension": 1024, "context_length": 8192},
        "databricks-bge-large-en": {"embedding_dimension": 1024, "context_length": 512},
    }

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    async def list_provider_model_ids(self) -> Iterable[str]:
        # Filter out None values from endpoint names
        api_token = self._get_api_key_from_config_or_provider_data()
        # WorkspaceClient expects base host without /serving-endpoints suffix
        base_url_str = str(self.config.base_url)
        if base_url_str.endswith("/serving-endpoints"):
            host = base_url_str[:-18]  # Remove '/serving-endpoints'
        else:
            host = base_url_str
        return [
            endpoint.name  # type: ignore[misc]
            for endpoint in WorkspaceClient(
                host=host, token=api_token
            ).serving_endpoints.list()  # TODO: this is not async
        ]

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        raise NotImplementedError()
