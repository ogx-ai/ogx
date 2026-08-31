# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub optional provider dependencies so these tests run without the heavy
# backend packages installed (same pattern as tests/unit/providers/test_milvus_weights.py).
if "pymilvus" not in sys.modules:
    pymilvus = ModuleType("pymilvus")
    pymilvus.AnnSearchRequest = object
    pymilvus.DataType = SimpleNamespace(
        VARCHAR="VARCHAR",
        FLOAT_VECTOR="FLOAT_VECTOR",
        JSON="JSON",
        SPARSE_FLOAT_VECTOR="SPARSE_FLOAT_VECTOR",
    )
    pymilvus.Function = object
    pymilvus.FunctionType = SimpleNamespace(BM25="BM25")
    pymilvus.MilvusClient = object
    pymilvus.RRFRanker = object
    pymilvus.WeightedRanker = object
    sys.modules["pymilvus"] = pymilvus

if "chromadb" not in sys.modules:
    chromadb = MagicMock(name="chromadb")
    chromadb.AsyncHttpClient = AsyncMock()
    sys.modules["chromadb"] = chromadb

if "weaviate" not in sys.modules:
    weaviate = MagicMock(name="weaviate")
    sys.modules["weaviate"] = weaviate
    sys.modules["weaviate.classes"] = weaviate.classes
    sys.modules["weaviate.classes.init"] = weaviate.classes.init
    sys.modules["weaviate.classes.query"] = weaviate.classes.query

from ogx.providers.inline.vector_io.milvus import MilvusVectorIOConfig as InlineMilvusVectorIOConfig
from ogx.providers.remote.vector_io.chroma import chroma as chroma_module
from ogx.providers.remote.vector_io.chroma.chroma import ChromaVectorIOAdapter
from ogx.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig as RemoteChromaVectorIOConfig
from ogx.providers.remote.vector_io.milvus import milvus as milvus_module
from ogx.providers.remote.vector_io.milvus.milvus import MilvusVectorIOAdapter
from ogx.providers.remote.vector_io.weaviate import weaviate as weaviate_module
from ogx.providers.remote.vector_io.weaviate.config import WeaviateVectorIOConfig
from ogx.providers.remote.vector_io.weaviate.weaviate import WeaviateVectorIOAdapter
from ogx_api.vector_stores import VectorStore


def _vector_store(store_id: str) -> VectorStore:
    return VectorStore(
        identifier=store_id,
        provider_id="test-provider",
        embedding_model="test-embedding-model",
        embedding_dimension=8,
    )


async def _make_milvus_adapter(kvstore_config, tmp_path, monkeypatch):
    fake_client = MagicMock()
    fake_client.has_collection.return_value = True
    monkeypatch.setattr(milvus_module, "MilvusClient", lambda **kwargs: fake_client)
    config = InlineMilvusVectorIOConfig(
        db_path=str(tmp_path / "milvus.db"),
        persistence=kvstore_config,
    )
    adapter = MilvusVectorIOAdapter(config, inference_api=MagicMock(), files_api=None)
    await adapter.initialize()
    return adapter


async def _make_chroma_adapter(kvstore_config, tmp_path, monkeypatch):
    # Always fake the Chroma HTTP client: with chromadb installed locally the
    # module stub above is skipped, and a real client would try to reach
    # localhost:8000 during initialize(), making the test environment-dependent.
    fake_client = MagicMock()
    fake_client.get_or_create_collection = AsyncMock(
        side_effect=lambda name, metadata=None: SimpleNamespace(name=name, metadata=metadata)
    )
    fake_client.get_collection = AsyncMock(side_effect=lambda name: SimpleNamespace(name=name))
    fake_client.delete_collection = AsyncMock()
    monkeypatch.setattr(chroma_module.chromadb, "AsyncHttpClient", AsyncMock(return_value=fake_client))
    config = RemoteChromaVectorIOConfig(
        url="http://localhost:8000",
        persistence=kvstore_config,
    )
    adapter = ChromaVectorIOAdapter(config, inference_api=MagicMock(), files_api=None)
    await adapter.initialize()
    return adapter


async def _make_weaviate_adapter(kvstore_config, tmp_path, monkeypatch):
    config = WeaviateVectorIOConfig(
        weaviate_cluster_url="localhost:8080",
        persistence=kvstore_config,
    )
    adapter = WeaviateVectorIOAdapter(config, inference_api=MagicMock(), files_api=None)
    fake_client = MagicMock()
    monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
    await adapter.initialize()
    return adapter


@pytest.mark.parametrize(
    "make_adapter,prefix_attr",
    [
        (_make_milvus_adapter, (milvus_module, "VECTOR_DBS_PREFIX")),
        (_make_chroma_adapter, (chroma_module, "VECTOR_DBS_PREFIX")),
        (_make_weaviate_adapter, (weaviate_module, "VECTOR_DBS_PREFIX")),
    ],
    ids=["milvus", "chroma", "weaviate"],
)
async def test_register_vector_store_persists_metadata(
    make_adapter, prefix_attr, unique_kvstore_config, tmp_path, monkeypatch
):
    module, attr = prefix_attr
    adapter = await make_adapter(unique_kvstore_config, tmp_path, monkeypatch)
    vector_store = _vector_store("vs-persist")

    await adapter.register_vector_store(vector_store)

    raw = await adapter.kvstore.get(f"{getattr(module, attr)}vs-persist")
    assert raw is not None
    persisted = VectorStore.model_validate_json(raw)
    assert persisted.identifier == vector_store.identifier
    assert persisted.embedding_model == vector_store.embedding_model
    assert persisted.embedding_dimension == vector_store.embedding_dimension


@pytest.mark.parametrize(
    "make_adapter",
    [_make_milvus_adapter, _make_chroma_adapter, _make_weaviate_adapter],
    ids=["milvus", "chroma", "weaviate"],
)
async def test_vector_store_survives_restart(make_adapter, unique_kvstore_config, tmp_path, monkeypatch):
    adapter = await make_adapter(unique_kvstore_config, tmp_path, monkeypatch)
    vector_store = _vector_store("vs-restart")
    await adapter.register_vector_store(vector_store)

    # Simulate a server restart: a brand new adapter backed by the same kvstore
    restarted = await make_adapter(unique_kvstore_config, tmp_path, monkeypatch)

    index = await restarted._get_and_cache_vector_store_index("vs-restart")
    assert index is not None
    assert index.vector_store.identifier == "vs-restart"


@pytest.mark.parametrize(
    "make_adapter,prefix_attr",
    [
        (_make_milvus_adapter, (milvus_module, "VECTOR_DBS_PREFIX")),
        (_make_chroma_adapter, (chroma_module, "VECTOR_DBS_PREFIX")),
        (_make_weaviate_adapter, (weaviate_module, "VECTOR_DBS_PREFIX")),
    ],
    ids=["milvus", "chroma", "weaviate"],
)
async def test_unregister_vector_store_removes_metadata(
    make_adapter, prefix_attr, unique_kvstore_config, tmp_path, monkeypatch
):
    module, attr = prefix_attr
    adapter = await make_adapter(unique_kvstore_config, tmp_path, monkeypatch)
    vector_store = _vector_store("vs-unregister")
    await adapter.register_vector_store(vector_store)

    await adapter.unregister_vector_store("vs-unregister")

    assert await adapter.kvstore.get(f"{getattr(module, attr)}vs-unregister") is None
    assert "vs-unregister" not in adapter.cache
