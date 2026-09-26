# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import ssl
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


class TestRerankUsesNetworkConfig:
    """rerank() builds its own httpx client, which must honour config.network like the OpenAI client."""

    @staticmethod
    async def _rerank_client_kwargs(adapter: LlamaCppServerInferenceAdapter) -> dict:
        with patch("httpx.AsyncClient") as mock_client_class:
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.9}]}
            client = MagicMock()
            client.post = AsyncMock(return_value=response)
            mock_client_class.return_value.__aenter__.return_value = client

            await adapter.rerank(RerankRequest(model="bge-reranker-v2-m3", query="q", items=["doc"]))

        mock_client_class.assert_called_once()
        return mock_client_class.call_args.kwargs

    async def test_applies_network_config(self):
        adapter = _make_adapter(
            network={
                "tls": {"verify": False},
                "proxy": {"url": "http://proxy.example.com:3128"},
                "headers": {"X-Route": "team-a"},
                "timeout": 12.0,
                "limits": {"max_connections": 7},
            }
        )

        kwargs = await self._rerank_client_kwargs(adapter)

        assert kwargs["verify"] is False
        assert set(kwargs["mounts"]) == {"http://", "https://"}
        assert kwargs["headers"] == {"X-Route": "team-a"}
        assert kwargs["timeout"] == httpx.Timeout(12.0)
        assert kwargs["limits"].max_connections == 7

    async def test_keeps_the_shared_ssl_context_when_only_a_proxy_is_configured(self):
        adapter = _make_adapter(network={"proxy": {"url": "http://proxy.example.com:3128"}})

        kwargs = await self._rerank_client_kwargs(adapter)

        assert set(kwargs["mounts"]) == {"http://", "https://"}
        assert kwargs["verify"] is adapter.shared_ssl_context

    async def test_uses_shared_ssl_context_without_network_config(self):
        adapter = _make_adapter()

        kwargs = await self._rerank_client_kwargs(adapter)

        assert kwargs == {"verify": adapter.shared_ssl_context}
        assert isinstance(kwargs["verify"], ssl.SSLContext)


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


NETWORK_CONFIG = {
    "tls": {"verify": False},
    "proxy": {"url": "http://proxy.example.com:3128"},
    "headers": {"X-Route": "team-a"},
    "timeout": 120.0,
    "limits": {"max_connections": 7},
}


class TestSignatureProbesUseNetworkConfig:
    """initialize() and health() probe /props on the configured server, so they must honour config.network."""

    @staticmethod
    def _capture(monkeypatch):
        real_client = httpx.AsyncClient
        captured: list[dict] = []

        def factory(*args, **kwargs):
            captured.append(kwargs)
            return real_client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"default_generation_settings": {}, "total_slots": 1})
                )
            )

        monkeypatch.setattr(server_signature.httpx, "AsyncClient", factory)
        return captured

    @pytest.mark.parametrize("probe", ["initialize", "health"])
    async def test_probe_applies_network_config(self, monkeypatch, probe):
        captured = self._capture(monkeypatch)

        await getattr(_make_adapter(network=NETWORK_CONFIG), probe)()

        kwargs = captured[0]
        assert kwargs["verify"] is False
        assert set(kwargs["mounts"]) == {"http://", "https://"}
        assert kwargs["headers"] == {"X-Route": "team-a"}
        assert kwargs["limits"].max_connections == 7
        assert kwargs["timeout"] == server_signature.SIGNATURE_TIMEOUT

    @pytest.mark.parametrize("probe", ["initialize", "health"])
    async def test_probe_uses_shared_ssl_context_without_network_config(self, monkeypatch, probe):
        captured = self._capture(monkeypatch)
        adapter = _make_adapter()

        await getattr(adapter, probe)()

        assert captured[0]["verify"] is adapter.shared_ssl_context
