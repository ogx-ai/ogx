# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from ogx_api import Model, Models, ModelType
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
    impl.get_model.return_value = Model(
        identifier="test-provider/test-model",
        provider_resource_id="test-model",
        provider_id="test-provider",
        model_type=ModelType.llm,
    )

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1/models/{model_id:path}", "GET")
    response = await get_endpoint(
        model_request=GetModelRequest(model_id="models/test-provider/test-model"),
        anthropic_version=None,
        x_goog_api_key="test-api-key",
    )

    impl.get_model.assert_awaited_once()
    called_request = impl.get_model.call_args.args[0]
    assert isinstance(called_request, GetModelRequest)
    assert called_request.model_id == "test-provider/test-model"

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["name"] == "models/test-provider/test-model"
