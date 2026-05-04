# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from ogx_api import ListModelsResponse, Model, Models, ModelType
from ogx_api.models import GetModelRequest
from ogx_api.models.fastapi_routes import create_router

# Mark all async tests in this module to use anyio with asyncio backend only
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _get_endpoint(router, path: str, method: str = "GET"):
    return next(
        r.endpoint for r in router.routes if getattr(r, "path", None) == path and method in getattr(r, "methods", set())
    )


async def test_google_get_model_normalizes_models_prefix():
    impl = AsyncMock(spec=Models)
    model = Model(
        identifier="test-provider/test-model",
        provider_resource_id="test-model",
        provider_id="test-provider",
        model_type=ModelType.llm,
    )
    impl.list_models.return_value = ListModelsResponse(data=[model])
    impl.get_model.return_value = model

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1/models/{model_id:path}", "GET")
    response = await get_endpoint(
        model_request=GetModelRequest(model_id="models/test-provider/test-model"),
        anthropic_version=None,
        x_goog_api_key="test-api-key",
        x_goog_user_project=None,
        x_goog_api_client=None,
    )

    impl.list_models.assert_awaited_once()
    impl.get_model.assert_awaited_once()
    called_request = impl.get_model.call_args.args[0]
    assert isinstance(called_request, GetModelRequest)
    assert called_request.model_id == "test-provider/test-model"

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["name"] == "models/test-provider/test-model"


async def test_google_get_model_resolves_vertex_resource_name():
    impl = AsyncMock(spec=Models)
    model = Model(
        identifier="vertexai/publishers/google/models/gemini-2.5-flash",
        provider_resource_id="publishers/google/models/gemini-2.5-flash",
        provider_id="vertexai",
        model_type=ModelType.llm,
    )
    impl.list_models.return_value = ListModelsResponse(data=[model])
    impl.get_model.return_value = model

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1/models/{model_id:path}", "GET")
    response = await get_endpoint(
        model_request=GetModelRequest(model_id="publishers/google/models/gemini-2.5-flash"),
        anthropic_version=None,
        x_goog_api_key=None,
        x_goog_user_project="my-project",
        x_goog_api_client=None,
    )

    impl.list_models.assert_awaited_once()
    impl.get_model.assert_awaited_once()
    called_request = impl.get_model.call_args.args[0]
    assert isinstance(called_request, GetModelRequest)
    assert called_request.model_id == "vertexai/publishers/google/models/gemini-2.5-flash"

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["name"] == "models/vertexai/publishers/google/models/gemini-2.5-flash"


async def test_google_get_model_rejects_ambiguous_provider_resource_id():
    impl = AsyncMock(spec=Models)
    impl.list_models.return_value = ListModelsResponse(
        data=[
            Model(
                identifier="gemini/publishers/google/models/gemini-2.5-flash",
                provider_resource_id="publishers/google/models/gemini-2.5-flash",
                provider_id="gemini",
                model_type=ModelType.llm,
            ),
            Model(
                identifier="vertexai/publishers/google/models/gemini-2.5-flash",
                provider_resource_id="publishers/google/models/gemini-2.5-flash",
                provider_id="vertexai",
                model_type=ModelType.llm,
            ),
        ]
    )

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1/models/{model_id:path}", "GET")
    with pytest.raises(ValueError, match="Failed to get model: Google model ID is ambiguous across providers"):
        await get_endpoint(
            model_request=GetModelRequest(model_id="publishers/google/models/gemini-2.5-flash"),
            anthropic_version=None,
            x_goog_api_key=None,
            x_goog_user_project="my-project",
            x_goog_api_client=None,
        )
    impl.get_model.assert_not_called()
