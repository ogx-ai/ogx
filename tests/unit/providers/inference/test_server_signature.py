# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx
import pytest

from ogx.providers.utils.inference import server_signature


def _install_mock_transport(monkeypatch, handler):
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(server_signature.httpx, "AsyncClient", factory)


class TestVerifyTextEmbeddingsInferenceServer:
    async def test_passes_and_strips_v1_suffix(self, monkeypatch):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"model_id": "BAAI/bge-small-en-v1.5", "dtype": "float32"})

        _install_mock_transport(monkeypatch, handler)
        await server_signature.verify_text_embeddings_inference_server("http://localhost:8080/v1")

        assert str(seen[0].url) == "http://localhost:8080/info"

    async def test_rejects_server_without_info_endpoint(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        _install_mock_transport(monkeypatch, handler)
        with pytest.raises(ValueError, match="Failed to verify"):
            await server_signature.verify_text_embeddings_inference_server("http://localhost:8080/v1")

    async def test_rejects_json_without_model_id(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"total_slots": 1})

        _install_mock_transport(monkeypatch, handler)
        with pytest.raises(ValueError, match="missing one of"):
            await server_signature.verify_text_embeddings_inference_server("http://localhost:8080/v1")

    async def test_rejects_non_json_body(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="OK")

        _install_mock_transport(monkeypatch, handler)
        with pytest.raises(ValueError, match="did not return JSON"):
            await server_signature.verify_text_embeddings_inference_server("http://localhost:8080/v1")


class TestGetTextEmbeddingsInferenceModelId:
    async def test_returns_model_id_and_strips_v1_suffix(self, monkeypatch):
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
        model_id = await server_signature.get_text_embeddings_inference_model_id("http://localhost:8080/v1")

        assert model_id == "nomic-ai/nomic-embed-text-v1.5"
        assert str(seen[0].url) == "http://localhost:8080/info"

    async def test_rejects_non_200_response(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        _install_mock_transport(monkeypatch, handler)
        with pytest.raises(ValueError, match="Failed to query"):
            await server_signature.get_text_embeddings_inference_model_id("http://localhost:8080/v1")

    async def test_rejects_json_without_model_id(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"total_slots": 1})

        _install_mock_transport(monkeypatch, handler)
        with pytest.raises(ValueError, match="missing model_id"):
            await server_signature.get_text_embeddings_inference_model_id("http://localhost:8080/v1")

    async def test_rejects_non_json_body(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="OK")

        _install_mock_transport(monkeypatch, handler)
        with pytest.raises(ValueError, match="did not return JSON"):
            await server_signature.get_text_embeddings_inference_model_id("http://localhost:8080/v1")

    async def test_connection_error_raises_server_unreachable_error(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        _install_mock_transport(monkeypatch, handler)
        with pytest.raises(server_signature.ServerUnreachableError, match="Failed to query"):
            await server_signature.get_text_embeddings_inference_model_id("http://localhost:8080/v1")

    async def test_api_key_sent_as_bearer_token(self, monkeypatch):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"model_id": "nomic-ai/nomic-embed-text-v1.5"})

        _install_mock_transport(monkeypatch, handler)
        await server_signature.get_text_embeddings_inference_model_id("http://localhost:8080/v1", api_key="sekret")

        assert seen[0].headers["Authorization"] == "Bearer sekret"


class TestVerifyLlamaCppServer:
    async def test_passes_with_total_slots(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"default_generation_settings": {}, "total_slots": 1})

        _install_mock_transport(monkeypatch, handler)
        await server_signature.verify_llama_cpp_server("http://localhost:8080/v1")

    async def test_passes_with_legacy_slot_keys(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"n_slots": 2})

        _install_mock_transport(monkeypatch, handler)
        await server_signature.verify_llama_cpp_server("http://localhost:8080/v1")

    async def test_rejects_server_without_props_endpoint(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        _install_mock_transport(monkeypatch, handler)
        with pytest.raises(ValueError, match="Failed to verify"):
            await server_signature.verify_llama_cpp_server("http://localhost:8080/v1")

    async def test_rejects_non_llama_cpp_props_body(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"model_id": "BAAI/bge-small-en-v1.5"})

        _install_mock_transport(monkeypatch, handler)
        with pytest.raises(ValueError, match="missing one of"):
            await server_signature.verify_llama_cpp_server("http://localhost:8080/v1")


class TestVerifyServerSignature:
    async def test_connection_error_raises_server_unreachable_error(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        _install_mock_transport(monkeypatch, handler)
        with pytest.raises(server_signature.ServerUnreachableError, match="Failed to verify"):
            await server_signature.verify_text_embeddings_inference_server("http://localhost:8080/v1")

    async def test_server_unreachable_error_is_value_error(self):
        assert issubclass(server_signature.ServerUnreachableError, ValueError)

    async def test_check_reports_unreachable_server_as_error(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        _install_mock_transport(monkeypatch, handler)
        result = await server_signature.check_text_embeddings_inference_server("http://localhost:8080/v1")

        assert result["status"] == server_signature.HealthStatus.ERROR
        assert "Failed to verify" in result["message"]

    async def test_api_key_sent_as_bearer_token(self, monkeypatch):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"model_id": "BAAI/bge-small-en-v1.5"})

        _install_mock_transport(monkeypatch, handler)
        await server_signature.verify_text_embeddings_inference_server("http://localhost:8080/v1", api_key="sekret")

        assert seen[0].headers["Authorization"] == "Bearer sekret"

    async def test_no_auth_header_without_api_key(self, monkeypatch):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"model_id": "BAAI/bge-small-en-v1.5"})

        _install_mock_transport(monkeypatch, handler)
        await server_signature.verify_text_embeddings_inference_server("http://localhost:8080/v1")

        assert "Authorization" not in seen[0].headers

    async def test_base_url_without_v1_suffix(self, monkeypatch):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"model_id": "BAAI/bge-small-en-v1.5"})

        _install_mock_transport(monkeypatch, handler)
        await server_signature.verify_text_embeddings_inference_server("http://localhost:8080")

        assert str(seen[0].url) == "http://localhost:8080/info"

    async def test_base_url_with_v1_and_trailing_slash(self, monkeypatch):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"model_id": "BAAI/bge-small-en-v1.5"})

        _install_mock_transport(monkeypatch, handler)
        await server_signature.verify_text_embeddings_inference_server("http://localhost:8080/v1/")

        assert str(seen[0].url) == "http://localhost:8080/info"


class TestRootUrl:
    def test_strips_v1(self):
        assert server_signature._root_url("http://localhost:8080/v1") == "http://localhost:8080"

    def test_strips_v1_and_trailing_slash(self):
        assert server_signature._root_url("http://localhost:8080/v1/") == "http://localhost:8080"

    def test_strips_trailing_slash(self):
        assert server_signature._root_url("http://localhost:8080/") == "http://localhost:8080"

    def test_keeps_subpath_prefix(self):
        assert server_signature._root_url("http://proxy.example/tei/v1") == "http://proxy.example/tei"
