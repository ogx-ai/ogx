# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from ogx.providers.remote.inference.llama_cpp_server.config import LlamaCppServerConfig
from ogx.providers.remote.inference.llama_cpp_server.llama_cpp_server import (
    LlamaCppServerInferenceAdapter,
)
from ogx.providers.utils.inference import server_signature
from ogx.providers.utils.inference.server_signature import ServerUnreachableError
from ogx_api import HealthStatus, ModelType, OpenAIChatCompletionContentPartImageParam
from ogx_api.inference import RerankRequest


def _make_adapter(**config_kwargs) -> LlamaCppServerInferenceAdapter:
    config = LlamaCppServerConfig(base_url="http://mocked.localhost:8080/v1", **config_kwargs)
    adapter = LlamaCppServerInferenceAdapter(config=config)
    adapter.__provider_id__ = "llama-cpp-server"
    return adapter


class TestConstructModelFromIdentifier:
    """construct_model_from_identifier() delegates to the shared classify_model();
    see test_models_dev_registry.py for classification coverage (models.dev lookup,
    metadata enrichment, name-heuristic fallback, rerank, HuggingFace precedence)."""

    def test_classified_model_uses_adapters_provider_id(self):
        adapter = _make_adapter()
        model = adapter.construct_model_from_identifier("nomic-embed-text-v1.5")

        assert model.model_type == ModelType.embedding
        assert model.provider_id == "llama-cpp-server"

    def test_unclassified_identifier_falls_through_to_default(self):
        adapter = _make_adapter()
        model = adapter.construct_model_from_identifier("qwen3-0.6b")

        assert model.model_type == ModelType.llm


def _mock_httpx_client(mock_client_class, results: list[dict]):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": results}
    mock_client_instance = MagicMock()
    mock_client_instance.post = AsyncMock(return_value=mock_response)
    mock_client_class.return_value.__aenter__.return_value = mock_client_instance
    return mock_client_instance


class TestRerank:
    """rerank() must hit /v1/rerank (base_url already includes /v1), sort by score,
    reject image inputs, and follow the same auth conventions as chat/embeddings (#6611)."""

    async def test_rerank_posts_to_v1_rerank_endpoint(self):
        adapter = _make_adapter()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = _mock_httpx_client(mock_client_class, [{"index": 0, "relevance_score": 0.9}])

            request = RerankRequest(model="bge-reranker-v2-m3", query="test", items=["doc1"])
            await adapter.rerank(request)

            call_args = mock_client_instance.post.call_args
            assert call_args.args[0] == "http://mocked.localhost:8080/v1/rerank"

    async def test_rerank_sends_auth_header(self):
        adapter = _make_adapter(api_key="my-secret-token")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = _mock_httpx_client(mock_client_class, [{"index": 0, "relevance_score": 0.9}])

            request = RerankRequest(model="bge-reranker-v2-m3", query="test", items=["doc1"])
            await adapter.rerank(request)

            headers = mock_client_instance.post.call_args.kwargs.get("headers", {})
            assert headers.get("Authorization") == "Bearer my-secret-token"

    async def test_rerank_no_auth_header_without_key(self):
        adapter = _make_adapter()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = _mock_httpx_client(mock_client_class, [{"index": 0, "relevance_score": 0.9}])

            request = RerankRequest(model="bge-reranker-v2-m3", query="test", items=["doc1"])
            await adapter.rerank(request)

            headers = mock_client_instance.post.call_args.kwargs.get("headers", {})
            assert "Authorization" not in headers

    async def test_rerank_sends_top_n_when_max_num_results_set(self):
        adapter = _make_adapter()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = _mock_httpx_client(mock_client_class, [{"index": 0, "relevance_score": 0.9}])

            request = RerankRequest(model="bge-reranker-v2-m3", query="test", items=["doc1", "doc2"], max_num_results=1)
            await adapter.rerank(request)

            payload = mock_client_instance.post.call_args.kwargs.get("json", {})
            assert payload["top_n"] == 1

    async def test_rerank_sorts_results_descending_by_score(self):
        adapter = _make_adapter()

        with patch("httpx.AsyncClient") as mock_client_class:
            _mock_httpx_client(
                mock_client_class,
                [
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ],
            )

            request = RerankRequest(model="bge-reranker-v2-m3", query="test", items=["a", "b", "c"])
            response = await adapter.rerank(request)

            assert [d.index for d in response.data] == [1, 2, 0]
            assert [d.relevance_score for d in response.data] == [0.9, 0.5, 0.2]

    async def test_rerank_rejects_image_query(self):
        adapter = _make_adapter()
        image = OpenAIChatCompletionContentPartImageParam(image_url={"url": "https://example.com/image.jpg"})
        request = RerankRequest(model="bge-reranker-v2-m3", query=image, items=["doc1"])

        with pytest.raises(ValueError, match="does not support images"):
            await adapter.rerank(request)

    async def test_rerank_rejects_image_item(self):
        adapter = _make_adapter()
        image = OpenAIChatCompletionContentPartImageParam(image_url={"url": "https://example.com/image.jpg"})
        request = RerankRequest(model="bge-reranker-v2-m3", query="test", items=[image])

        with pytest.raises(ValueError, match="does not support images"):
            await adapter.rerank(request)

    async def test_rerank_raises_runtime_error_on_non_200(self):
        adapter = _make_adapter()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "internal error"
            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            request = RerankRequest(model="bge-reranker-v2-m3", query="test", items=["doc1"])
            with pytest.raises(RuntimeError, match="status 500"):
                await adapter.rerank(request)


def _install_mock_transport(monkeypatch, handler):
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(server_signature.httpx, "AsyncClient", factory)


class TestHealth:
    async def test_ok_for_llama_cpp_server(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"default_generation_settings": {}, "total_slots": 1})

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter()

        result = await adapter.health()

        assert result["status"] == HealthStatus.OK

    async def test_error_for_non_llama_cpp_server(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter()

        result = await adapter.health()

        assert result["status"] == HealthStatus.ERROR
        assert "Failed to verify" in result["message"]

    async def test_sends_auth_credential(self, monkeypatch):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"default_generation_settings": {}, "total_slots": 1})

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter(api_key=SecretStr("sekret"))

        await adapter.health()

        assert seen[0].headers["Authorization"] == "Bearer sekret"


class TestInitialize:
    async def test_ok_for_llama_cpp_server(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"default_generation_settings": {}, "total_slots": 1})

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter()

        await adapter.initialize()

    async def test_raises_for_non_llama_cpp_server(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter()

        with pytest.raises(ValueError, match="Failed to verify") as excinfo:
            await adapter.initialize()
        assert not isinstance(excinfo.value, ServerUnreachableError)

    async def test_unreachable_server_does_not_raise(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter()

        await adapter.initialize()
