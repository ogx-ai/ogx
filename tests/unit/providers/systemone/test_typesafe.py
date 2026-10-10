# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from pydantic import SecretStr

from ogx.providers.remote.systemone.typesafe.config import TypeSafeConfig
from ogx.providers.remote.systemone.typesafe.typesafe import TypeSafeSystemOneAdapter
from ogx_api import ModelType, SystemOneRequest


def _request() -> SystemOneRequest:
    return SystemOneRequest(
        model="jev-latest",
        state="Help! My payouts have been failing for 3 days.",
        questions={
            "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        },
    )


def _mock_response(json_body: dict, status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, json=json_body, request=httpx2.Request("POST", "https://api.typesafe.ai/"))


@pytest.fixture
def adapter():
    impl = TypeSafeSystemOneAdapter(TypeSafeConfig(api_key="test-key"))
    impl._client = MagicMock(spec=httpx2.AsyncClient)
    return impl


class TestJudge:
    async def test_posts_to_v1_systemone_with_the_request_body(self, adapter):
        response_body = {
            "model": "jev-1.13.0",
            "answers": {"is_urgent": {"type": "noul", "noul": 0.95}},
            "usage": {"input_tokens": 296, "output_tokens": 20},
        }
        adapter._client.post = AsyncMock(return_value=_mock_response(response_body))

        with patch.object(adapter, "get_request_provider_data", return_value=None):
            result = await adapter.judge(_request())

        call = adapter._client.post.call_args
        assert call.args[0] == "/v1/systemone"
        assert call.kwargs["json"]["model"] == "jev-latest"
        assert call.kwargs["json"]["questions"]["is_urgent"]["type"] == "noul"
        assert result.model == "jev-1.13.0"
        assert result.answers["is_urgent"].noul == 0.95

    async def test_sends_bearer_auth_header_from_config(self, adapter):
        adapter._client.post = AsyncMock(return_value=_mock_response({"model": "m", "answers": {}, "usage": {}}))

        with patch.object(adapter, "get_request_provider_data", return_value=None):
            await adapter.judge(_request())

        headers = adapter._client.post.call_args.kwargs["headers"]
        assert headers["authorization"] == "Bearer test-key"

    async def test_provider_data_api_key_overrides_config(self, adapter):
        adapter._client.post = AsyncMock(return_value=_mock_response({"model": "m", "answers": {}, "usage": {}}))

        with patch.object(
            adapter,
            "get_request_provider_data",
            return_value=MagicMock(typesafe_api_key=SecretStr("per-request-key")),
        ):
            await adapter.judge(_request())

        headers = adapter._client.post.call_args.kwargs["headers"]
        assert headers["authorization"] == "Bearer per-request-key"

    async def test_no_api_key_omits_authorization_header(self):
        impl = TypeSafeSystemOneAdapter(TypeSafeConfig())
        impl._client = MagicMock(spec=httpx2.AsyncClient)
        impl._client.post = AsyncMock(return_value=_mock_response({"model": "m", "answers": {}, "usage": {}}))

        with patch.object(impl, "get_request_provider_data", return_value=None):
            await impl.judge(_request())

        headers = impl._client.post.call_args.kwargs["headers"]
        assert "authorization" not in headers

    async def test_raises_before_initialization(self):
        impl = TypeSafeSystemOneAdapter(TypeSafeConfig())
        with pytest.raises(RuntimeError, match="not initialized"):
            await impl.judge(_request())


class TestListModels:
    async def test_lists_models_from_v1_models(self, adapter):
        adapter.__provider_id__ = "typesafe"
        adapter._client.get = AsyncMock(return_value=_mock_response({"data": [{"id": "jev-latest"}, {"id": "nimble"}]}))

        with patch.object(adapter, "get_request_provider_data", return_value=None):
            models = await adapter.list_models()

        assert adapter._client.get.call_args.args[0] == "/v1/models"
        assert models is not None
        assert [m.identifier for m in models] == ["jev-latest", "nimble"]
        assert all(m.model_type == ModelType.systemone for m in models)

    async def test_empty_models_list(self, adapter):
        adapter._client.get = AsyncMock(return_value=_mock_response({"data": []}))

        with patch.object(adapter, "get_request_provider_data", return_value=None):
            models = await adapter.list_models()

        assert models == []

    async def test_should_refresh_models_is_true(self, adapter):
        assert await adapter.should_refresh_models() is True


class TestRegisterUnregisterModel:
    async def test_register_model_returns_model_unchanged(self, adapter):
        model = MagicMock()
        assert await adapter.register_model(model) is model

    async def test_unregister_model_is_a_noop(self, adapter):
        assert await adapter.unregister_model("jev-latest") is None
