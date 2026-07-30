# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ogx.core.server.fastapi_router_registry import build_fastapi_router
from ogx_api import VectorStoreListResponse, VectorStoreSearchResponsePage
from ogx_api.datatypes import Api


def test_vector_io_router_list_vector_stores() -> None:
    impl = AsyncMock()
    impl.openai_list_vector_stores = AsyncMock(
        return_value=VectorStoreListResponse(data=[], first_id="", last_id="", has_more=False)
    )

    router = build_fastapi_router(Api.vector_io, impl)
    assert router is not None

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)
    response = client.get("/v1/vector_stores")

    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": [], "first_id": "", "last_id": "", "has_more": False}

    impl.openai_list_vector_stores.assert_awaited_once()


def test_vector_io_router_search_vector_store_passes_body_fields() -> None:
    impl = AsyncMock()
    impl.openai_search_vector_store = AsyncMock(
        return_value=VectorStoreSearchResponsePage(search_query=["hello"], data=[], has_more=False)
    )

    router = build_fastapi_router(Api.vector_io, impl)
    assert router is not None

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)
    response = client.post(
        "/v1/vector_stores/vs_123/search",
        json={"query": "hello", "rewrite_query": True},
    )

    assert response.status_code == 200
    assert response.json()["object"] == "vector_store.search_results.page"

    impl.openai_search_vector_store.assert_awaited_once()
    _, kwargs = impl.openai_search_vector_store.call_args
    assert kwargs["vector_store_id"] == "vs_123"
    request = kwargs["request"]
    assert request.query == "hello"
    assert request.rewrite_query is True
    assert request.search_mode == "vector"


def test_vector_io_router_openapi_tags_vector_stores_separately() -> None:
    impl = AsyncMock()

    router = build_fastapi_router(Api.vector_io, impl)
    assert router is not None

    app = FastAPI()
    app.include_router(router)

    schema = app.openapi()
    expected_tags = {
        ("post", "/v1/vector-io/insert"): ["VectorIO"],
        ("post", "/v1/vector-io/query"): ["VectorIO"],
        ("get", "/v1/vector_stores"): ["Vector Stores"],
        ("post", "/v1/vector_stores"): ["Vector Stores"],
        ("delete", "/v1/vector_stores/{vector_store_id}"): ["Vector Stores"],
        ("get", "/v1/vector_stores/{vector_store_id}"): ["Vector Stores"],
        ("post", "/v1/vector_stores/{vector_store_id}"): ["Vector Stores"],
        ("post", "/v1/vector_stores/{vector_store_id}/file_batches"): ["Vector Stores"],
        ("get", "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}"): ["Vector Stores"],
        ("post", "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel"): ["Vector Stores"],
        ("get", "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files"): ["Vector Stores"],
        ("get", "/v1/vector_stores/{vector_store_id}/files"): ["Vector Stores"],
        ("post", "/v1/vector_stores/{vector_store_id}/files"): ["Vector Stores"],
        ("delete", "/v1/vector_stores/{vector_store_id}/files/{file_id}"): ["Vector Stores"],
        ("get", "/v1/vector_stores/{vector_store_id}/files/{file_id}"): ["Vector Stores"],
        ("post", "/v1/vector_stores/{vector_store_id}/files/{file_id}"): ["Vector Stores"],
        ("get", "/v1/vector_stores/{vector_store_id}/files/{file_id}/content"): ["Vector Stores"],
        ("post", "/v1/vector_stores/{vector_store_id}/search"): ["Vector Stores"],
    }

    actual_tags = {
        (method, path): operation["tags"]
        for path, methods in schema["paths"].items()
        if path.startswith("/v1/vector")
        for method, operation in methods.items()
    }

    assert actual_tags == expected_tags
