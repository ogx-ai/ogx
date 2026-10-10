# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

import httpx2

from ogx.core.request_headers import NeedsRequestProviderData
from ogx.log import get_logger
from ogx_api import Model, ModelsProtocolPrivate, ModelType, SystemOne, SystemOneRequest, SystemOneResponse

from .config import TypeSafeConfig

logger = get_logger(name=__name__, category="systemone::typesafe")


class TypeSafeSystemOneAdapter(SystemOne, ModelsProtocolPrivate, NeedsRequestProviderData):
    """SystemOne judgment provider backed by TypeSafe's Jev API, the canonical implementation
    of the SystemOne wire format. Forwards /v1/systemone requests and responses as-is."""

    def __init__(self, config: TypeSafeConfig):
        self.config = config
        self._client: httpx2.AsyncClient | None = None

    async def initialize(self) -> None:
        self._client = httpx2.AsyncClient(base_url=str(self.config.base_url).rstrip("/"), timeout=30.0)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_api_key(self) -> str | None:
        api_key = self.config.api_key.get_secret_value() if self.config.api_key else None
        provider_data = self.get_request_provider_data()
        if provider_data and provider_data.typesafe_api_key:
            api_key = provider_data.typesafe_api_key.get_secret_value()
        return api_key

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        api_key = self._get_api_key()
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        return headers

    async def judge(self, request: SystemOneRequest) -> SystemOneResponse:
        if self._client is None:
            raise RuntimeError("Failed to judge: provider not initialized")
        response = await self._client.post(
            "/v1/systemone", json=request.model_dump(exclude_none=True), headers=self._headers()
        )
        response.raise_for_status()
        return SystemOneResponse(**response.json())

    async def register_model(self, model: Model) -> Model:
        return model

    async def unregister_model(self, model_id: str) -> None:
        return None

    async def list_models(self) -> list[Model] | None:
        """List decision models from TypeSafe's own GET /v1/models.

        TypeSafe's /v1/models response shape is not publicly documented beyond "also GET
        /v1/models" (see #6746); this assumes the OpenAI-style {"data": [{"id": ...}, ...]}
        shape every other backend in this wire-format family (Ollama, vLLM, llama.cpp) uses.
        """
        if self._client is None:
            raise RuntimeError("Failed to list models: provider not initialized")
        response = await self._client.get("/v1/models", headers=self._headers())
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return [
            Model(
                identifier=entry["id"],
                provider_resource_id=entry["id"],
                provider_id=getattr(self, "__provider_id__", "typesafe"),
                model_type=ModelType.systemone,
            )
            for entry in data.get("data", [])
        ]

    async def should_refresh_models(self) -> bool:
        return True
