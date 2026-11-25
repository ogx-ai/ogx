# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Every Responses route returns OpenAI's response object shape (#7080).

OpenAI's response object has never had an `input` field; a response's input is served by
GET /v1/responses/{id}/input_items. GET /v1/responses used to leak the stored input on every
item, so the same object had a different shape depending on which route returned it.
"""

import time
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx2
import pytest
from fastapi import FastAPI

from ogx.core.server.fastapi_router_registry import build_fastapi_router
from ogx.core.storage.datatypes import ResponsesStoreReference, SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.utils.responses.responses_store import ResponsesStore
from ogx_api import Api, OpenAIUserMessageParam, Responses
from ogx_api.openai_responses import (
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseOutputMessageContentOutputText,
)
from ogx_api.responses.models import ListResponsesRequest, RetrieveResponseRequest


@pytest.fixture
async def store(tmp_path) -> ResponsesStore:
    backend_name = f"sql_responses_{uuid4().hex}"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=str(tmp_path / "responses.db"))})
    store = ResponsesStore(ResponsesStoreReference(backend=backend_name, table_name="responses"), policy=[])
    await store.initialize()
    return store


async def _store_two_turns(store: ResponsesStore) -> None:
    """A response and an incremental follow-up, so both stored input modes are covered."""
    now = int(time.time())
    for index, previous_response_id in ((1, None), (2, "resp-shape-1")):
        response = OpenAIResponseObject(
            id=f"resp-shape-{index}",
            created_at=now + index,
            model="test-model",
            output=[
                OpenAIResponseMessage(
                    id=f"out-shape-{index}",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text=f"Answer {index}")],
                )
            ],
            status="completed",
            previous_response_id=previous_response_id,
            store=True,
        )
        await store.store_response_object(
            response,
            [OpenAIResponseMessage(id=f"in-shape-{index}", role="user", content=f"Question {index}")],
            [OpenAIUserMessageParam(content=f"Question {index}")],
            incremental_input=previous_response_id is not None,
        )
    await store.flush()


def _app(store: ResponsesStore) -> FastAPI:
    impl = AsyncMock(spec=Responses)

    async def list_openai_responses(request: ListResponsesRequest):
        return await store.list_responses(request.after, request.limit, request.model, request.order)

    async def get_openai_response(request: RetrieveResponseRequest):
        stored = await store.get_response_object(request.response_id, reconstruct_input=False)
        return stored.to_response_object()

    impl.list_openai_responses.side_effect = list_openai_responses
    impl.get_openai_response.side_effect = get_openai_response

    app = FastAPI()
    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)
    return app


async def _get(app: FastAPI, path: str) -> dict:
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == 200, response.text
    return response.json()


async def test_listed_responses_carry_no_input(store):
    await _store_two_turns(store)

    listed = await _get(_app(store), "/v1/responses")

    assert {item["id"] for item in listed["data"]} == {"resp-shape-1", "resp-shape-2"}
    for item in listed["data"]:
        assert "input" not in item
        assert "messages" not in item
        assert "input_storage_mode" not in item


async def test_listed_and_retrieved_responses_have_the_same_shape(store):
    await _store_two_turns(store)
    app = _app(store)

    listed = await _get(app, "/v1/responses")
    for item in listed["data"]:
        retrieved = await _get(app, f"/v1/responses/{item['id']}")
        assert item == retrieved


def test_openapi_list_items_are_plain_response_objects():
    app = FastAPI()
    router = build_fastapi_router(Api.responses, AsyncMock(spec=Responses))
    assert router is not None
    app.include_router(router)

    schemas = app.openapi()["components"]["schemas"]

    assert schemas["ListOpenAIResponseObject"]["properties"]["data"]["items"] == {
        "$ref": "#/components/schemas/OpenAIResponseObject"
    }
    assert "input" not in schemas["OpenAIResponseObject"]["properties"]
    assert "OpenAIResponseObjectWithInput" not in schemas
