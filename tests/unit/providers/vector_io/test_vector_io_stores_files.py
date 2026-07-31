# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from unittest.mock import AsyncMock, patch

import pytest

# This test is a unit test for the inline VectorIO providers. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_vector_io_openai_vector_stores.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto


@pytest.fixture(autouse=True)
def mock_resume_file_batches(request):
    """Mock the resume functionality to prevent stale file batches from being processed during tests."""
    with patch(
        "ogx.providers.utils.memory.openai_vector_store_mixin.OpenAIVectorStoreMixin._resume_incomplete_batches",
        new_callable=AsyncMock,
    ):
        yield


async def test_save_openai_vector_store(vector_io_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_store_id": "test_db",
        "embedding_model": "test_model",
    }

    await vector_io_adapter._save_openai_vector_store(store_id, openai_vector_store)

    assert openai_vector_store["id"] in vector_io_adapter.openai_vector_stores
    assert vector_io_adapter.openai_vector_stores[openai_vector_store["id"]] == openai_vector_store


async def test_update_openai_vector_store(vector_io_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_store_id": "test_db",
        "embedding_model": "test_model",
    }

    await vector_io_adapter._save_openai_vector_store(store_id, openai_vector_store)
    openai_vector_store["description"] = "Updated description"
    await vector_io_adapter._update_openai_vector_store(store_id, openai_vector_store)
    assert vector_io_adapter.openai_vector_stores[openai_vector_store["id"]] == openai_vector_store


async def test_delete_openai_vector_store(vector_io_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_store_id": "test_db",
        "embedding_model": "test_model",
    }

    await vector_io_adapter._save_openai_vector_store(store_id, openai_vector_store)
    await vector_io_adapter._delete_openai_vector_store_from_storage(store_id)
    assert openai_vector_store["id"] not in vector_io_adapter.openai_vector_stores


async def test_load_openai_vector_stores(vector_io_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_store_id": "test_db",
        "embedding_model": "test_model",
    }

    await vector_io_adapter._save_openai_vector_store(store_id, openai_vector_store)
    loaded_stores = await vector_io_adapter._load_openai_vector_stores()
    assert loaded_stores[store_id] == openai_vector_store


async def test_save_openai_vector_store_file(vector_io_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    # validating we don't raise an exception
    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)


async def test_update_openai_vector_store_file(vector_io_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)

    updated_file_info = file_info.copy()
    updated_file_info["filename"] = "updated_test_file.txt"

    await vector_io_adapter._update_openai_vector_store_file(
        store_id,
        file_id,
        updated_file_info,
    )

    loaded_contents = await vector_io_adapter._load_openai_vector_store_file(store_id, file_id)
    assert loaded_contents == updated_file_info
    assert loaded_contents != file_info


async def test_load_openai_vector_store_file_contents(vector_io_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)

    loaded_contents = await vector_io_adapter._load_openai_vector_store_file_contents(store_id, file_id)
    assert loaded_contents == file_contents


async def test_delete_openai_vector_store_file_from_storage(vector_io_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    await vector_io_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)
    await vector_io_adapter._delete_openai_vector_store_file_from_storage(store_id, file_id)

    loaded_file_info = await vector_io_adapter._load_openai_vector_store_file(store_id, file_id)
    assert loaded_file_info == {}
    loaded_contents = await vector_io_adapter._load_openai_vector_store_file_contents(store_id, file_id)
    assert loaded_contents == []
