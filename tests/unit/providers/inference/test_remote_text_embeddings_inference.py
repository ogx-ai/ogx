# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx
import pytest
from pydantic import SecretStr

from ogx.providers.remote.inference.text_embeddings_inference.config import TextEmbeddingsInferenceConfig
from ogx.providers.remote.inference.text_embeddings_inference.text_embeddings_inference import (
    TextEmbeddingsInferenceAdapter,
)
from ogx.providers.utils.inference import server_signature
from ogx.providers.utils.inference.server_signature import ServerUnreachableError
from ogx_api import (
    HealthStatus,
    ModelType,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletionRequestWithExtraBody,
)


def _make_adapter(**config_kwargs) -> TextEmbeddingsInferenceAdapter:
    config = TextEmbeddingsInferenceConfig(base_url="http://mocked.localhost:8080/v1", **config_kwargs)
    adapter = TextEmbeddingsInferenceAdapter(config=config)
    adapter.__provider_id__ = "text-embeddings-inference"
    return adapter


def _install_mock_transport(monkeypatch, handler):
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(server_signature.httpx, "AsyncClient", factory)


class TestHealth:
    async def test_ok_for_tei_server(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"model_id": "BAAI/bge-small-en-v1.5"})

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter()

        result = await adapter.health()

        assert result["status"] == HealthStatus.OK

    async def test_error_for_non_tei_server(self, monkeypatch):
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
            return httpx.Response(200, json={"model_id": "BAAI/bge-small-en-v1.5"})

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter(api_key=SecretStr("sekret"))

        await adapter.health()

        assert seen[0].headers["Authorization"] == "Bearer sekret"


class TestInitialize:
    async def test_ok_for_tei_server(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"model_id": "BAAI/bge-small-en-v1.5"})

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter()

        await adapter.initialize()

    async def test_raises_for_non_tei_server(self, monkeypatch):
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


class TestListProviderModelIds:
    async def test_lists_model_from_info_endpoint(self, monkeypatch):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200,
                json={
                    "model_id": "nomic-ai/nomic-embed-text-v1.5",
                    "served_model_name": "nomic-ai/nomic-embed-text-v1.5",
                },
            )

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter()

        model_ids = await adapter.list_provider_model_ids()

        assert model_ids == ["nomic-ai/nomic-embed-text-v1.5"]
        assert str(seen[0].url) == "http://mocked.localhost:8080/info"

    async def test_sends_auth_credential(self, monkeypatch):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"model_id": "nomic-ai/nomic-embed-text-v1.5"})

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter(api_key=SecretStr("sekret"))

        await adapter.list_provider_model_ids()

        assert seen[0].headers["Authorization"] == "Bearer sekret"

    async def test_propagates_unreachable_server(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        _install_mock_transport(monkeypatch, handler)
        adapter = _make_adapter()

        with pytest.raises(ServerUnreachableError, match="Failed to query"):
            await adapter.list_provider_model_ids()


class TestConstructModelFromIdentifier:
    def test_model_is_typed_as_embedding(self):
        adapter = _make_adapter()

        model = adapter.construct_model_from_identifier("nomic-ai/nomic-embed-text-v1.5")

        assert model.model_type == ModelType.embedding
        assert model.provider_id == "text-embeddings-inference"
        assert model.provider_resource_id == "nomic-ai/nomic-embed-text-v1.5"
        assert model.identifier == "nomic-ai/nomic-embed-text-v1.5"


class TestUnsupportedOperations:
    async def test_chat_completion_not_supported(self):
        adapter = _make_adapter()
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="text-embeddings-inference/nomic-ai/nomic-embed-text-v1.5",
            messages=[{"role": "user", "content": "Hello"}],
        )

        with pytest.raises(NotImplementedError, match="does not support chat completions"):
            await adapter.openai_chat_completion(params)

    async def test_completion_not_supported(self):
        adapter = _make_adapter()
        params = OpenAICompletionRequestWithExtraBody(
            model="text-embeddings-inference/nomic-ai/nomic-embed-text-v1.5",
            prompt="Hello",
        )

        with pytest.raises(NotImplementedError, match="does not support completions"):
            await adapter.openai_completion(params)


NETWORK_CONFIG = {
    "tls": {"verify": False},
    "proxy": {"url": "http://proxy.example.com:3128"},
    "headers": {"X-Route": "team-a"},
    "timeout": 120.0,
    "limits": {"max_connections": 7},
}


class TestSignatureProbesUseNetworkConfig:
    """initialize(), health() and model discovery call TEI's native /info, so they must honour config.network."""

    @staticmethod
    def _capture(monkeypatch):
        real_client = httpx.AsyncClient
        captured: list[dict] = []

        def factory(*args, **kwargs):
            captured.append(kwargs)
            return real_client(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"model_id": "BAAI/bge-small"}))
            )

        monkeypatch.setattr(server_signature.httpx, "AsyncClient", factory)
        return captured

    @pytest.mark.parametrize("probe", ["initialize", "health", "list_provider_model_ids"])
    async def test_probe_applies_network_config(self, monkeypatch, probe):
        captured = self._capture(monkeypatch)

        await getattr(_make_adapter(network=NETWORK_CONFIG), probe)()

        kwargs = captured[0]
        assert kwargs["verify"] is False
        assert set(kwargs["mounts"]) == {"http://", "https://"}
        assert kwargs["headers"] == {"X-Route": "team-a"}
        assert kwargs["limits"].max_connections == 7
        assert kwargs["timeout"] == server_signature.SIGNATURE_TIMEOUT

    @pytest.mark.parametrize("probe", ["initialize", "health", "list_provider_model_ids"])
    async def test_probe_uses_shared_ssl_context_without_network_config(self, monkeypatch, probe):
        captured = self._capture(monkeypatch)
        adapter = _make_adapter()

        await getattr(adapter, probe)()

        assert captured[0]["verify"] is adapter.shared_ssl_context
