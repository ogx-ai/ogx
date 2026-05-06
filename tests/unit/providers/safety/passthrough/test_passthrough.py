# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock

import httpx
import pytest

from ogx.providers.remote.safety.passthrough.config import PassthroughSafetyConfig
from ogx_api import (
    GetShieldRequest,
    OpenAIUserMessageParam,
    ResourceType,
    RunModerationRequest,
    RunShieldRequest,
    RunShieldResponse,
    Shield,
    ViolationLevel,
)

from .conftest import (
    FakePassthroughSafetyAdapter,
    _stub_provider_spec,
    mock_httpx_response,
    provider_data_ctx,
)


def _make_shield(shield_id: str = "test-shield", provider_resource_id: str = "text-moderation-latest") -> Shield:
    return Shield(
        provider_id="passthrough",
        type=ResourceType.shield,
        identifier=shield_id,
        provider_resource_id=provider_resource_id,
    )


# -- run_shield --


async def test_run_shield_safe_content(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield()
    downstream = {
        "id": "modr-123",
        "model": "text-moderation-latest",
        "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    result = await adapter.run_shield(
        RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="Hello")])
    )

    assert isinstance(result, RunShieldResponse)
    assert result.violation is None
    adapter.shield_store.get_shield.assert_called_once_with(GetShieldRequest(identifier="test-shield"))


async def test_run_shield_flagged_content(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield()
    downstream = {
        "id": "modr-456",
        "model": "text-moderation-latest",
        "results": [
            {
                "flagged": True,
                "categories": {"hate": True, "violence": False},
                "category_scores": {"hate": 0.95, "violence": 0.01},
            }
        ],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    result = await adapter.run_shield(
        RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="hateful")])
    )

    assert result.violation is not None
    assert result.violation.violation_level == ViolationLevel.ERROR
    assert result.violation.metadata["violation_type"] == "hate"


async def test_run_shield_not_found(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = None
    with pytest.raises(ValueError, match="not found"):
        await adapter.run_shield(
            RunShieldRequest(shield_id="nonexistent", messages=[OpenAIUserMessageParam(content="test")])
        )


async def test_run_shield_http_5xx(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield()
    resp = mock_httpx_response({}, status_code=500)
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error",
        request=MagicMock(spec=httpx.Request),
        response=resp,
    )

    adapter._client.post.return_value = resp
    with pytest.raises(RuntimeError, match="returned HTTP 500"):
        await adapter.run_shield(
            RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="test")])
        )


async def test_run_shield_http_4xx_raises_value_error(adapter: FakePassthroughSafetyAdapter) -> None:
    """4xx from downstream maps to ValueError (caller's fault), not RuntimeError."""
    adapter.shield_store.get_shield.return_value = _make_shield()
    resp = mock_httpx_response({}, status_code=400)
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request",
        request=MagicMock(spec=httpx.Request),
        response=resp,
    )

    adapter._client.post.return_value = resp
    with pytest.raises(ValueError, match="rejected the request"):
        await adapter.run_shield(
            RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="test")])
        )


async def test_run_shield_uses_provider_resource_id(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield(provider_resource_id="my-custom-model")
    downstream = {
        "id": "modr-789",
        "model": "my-custom-model",
        "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    await adapter.run_shield(
        RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="hello")])
    )

    sent_payload = adapter._client.post.call_args.kwargs.get("json") or adapter._client.post.call_args[1].get("json")
    assert sent_payload["model"] == "my-custom-model"


async def test_run_shield_url_construction(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield()
    downstream = {
        "id": "modr-abc",
        "model": "text-moderation-latest",
        "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    await adapter.run_shield(
        RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="hello")])
    )

    url = (
        adapter._client.post.call_args.args[0]
        if adapter._client.post.call_args.args
        else adapter._client.post.call_args.kwargs.get("url")
    )
    assert url == "https://safety.example.com/v1/moderations"


async def test_run_shield_multiple_messages(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield()
    downstream = {
        "id": "modr-multi",
        "model": "text-moderation-latest",
        "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    await adapter.run_shield(
        RunShieldRequest(
            shield_id="test-shield",
            messages=[
                OpenAIUserMessageParam(content="first"),
                OpenAIUserMessageParam(content="second"),
            ],
        )
    )

    sent_payload = adapter._client.post.call_args.kwargs.get("json") or adapter._client.post.call_args[1].get("json")
    assert sent_payload["input"] == ["first", "second"]


async def test_run_shield_empty_messages(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield()
    result = await adapter.run_shield(RunShieldRequest(shield_id="test-shield", messages=[]))
    assert result.violation is None


async def test_run_shield_empty_results_raises(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield()
    downstream = {"id": "modr-empty", "model": "text-moderation-latest", "results": []}

    adapter._client.post.return_value = mock_httpx_response(downstream)
    with pytest.raises(RuntimeError, match="empty results"):
        await adapter.run_shield(
            RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="hello")])
        )


# -- run_moderation --


async def test_run_moderation_safe(adapter: FakePassthroughSafetyAdapter) -> None:
    downstream = {
        "id": "modr-safe",
        "model": "text-moderation-latest",
        "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    result = await adapter.run_moderation(RunModerationRequest(input="safe text", model="text-moderation-latest"))

    assert len(result.results) == 1
    assert result.results[0].flagged is False
    assert result.model == "text-moderation-latest"


async def test_run_moderation_flagged(adapter: FakePassthroughSafetyAdapter) -> None:
    downstream = {
        "id": "modr-flagged",
        "model": "text-moderation-latest",
        "results": [{"flagged": True, "categories": {"violence": True}, "category_scores": {"violence": 0.99}}],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    result = await adapter.run_moderation(RunModerationRequest(input="violent", model="text-moderation-latest"))

    assert result.results[0].flagged is True
    assert result.results[0].categories == {"violence": True}


async def test_run_moderation_multiple_inputs(adapter: FakePassthroughSafetyAdapter) -> None:
    downstream = {
        "id": "modr-multi",
        "model": "text-moderation-latest",
        "results": [
            {"flagged": False, "categories": {}, "category_scores": {}},
            {"flagged": True, "categories": {"hate": True}, "category_scores": {"hate": 0.9}},
        ],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    result = await adapter.run_moderation(
        RunModerationRequest(input=["safe", "hateful"], model="text-moderation-latest")
    )

    assert len(result.results) == 2
    assert result.results[0].flagged is False
    assert result.results[1].flagged is True


async def test_run_moderation_no_model(adapter: FakePassthroughSafetyAdapter) -> None:
    downstream = {
        "id": "modr-nomodel",
        "model": "default-model",
        "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    result = await adapter.run_moderation(RunModerationRequest(input="test", model=None))

    sent_payload = adapter._client.post.call_args.kwargs.get("json") or adapter._client.post.call_args[1].get("json")
    assert "model" not in sent_payload
    assert result.model == "default-model"


async def test_run_moderation_http_error(adapter: FakePassthroughSafetyAdapter) -> None:
    resp = mock_httpx_response({}, status_code=502)
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error",
        request=MagicMock(spec=httpx.Request),
        response=resp,
    )

    adapter._client.post.return_value = resp
    with pytest.raises(RuntimeError, match="returned HTTP 502"):
        await adapter.run_moderation(RunModerationRequest(input="test", model="text-moderation-latest"))


# -- API key precedence --


async def test_api_key_from_config(adapter_with_api_key: FakePassthroughSafetyAdapter) -> None:
    assert adapter_with_api_key._get_api_key() == "config-key-123"


async def test_api_key_from_provider_data(adapter: FakePassthroughSafetyAdapter) -> None:
    with provider_data_ctx({"passthrough_api_key": "pd-key-456"}):
        assert adapter._get_api_key() == "pd-key-456"


async def test_api_key_config_wins_over_provider_data(adapter_with_api_key: FakePassthroughSafetyAdapter) -> None:
    with provider_data_ctx({"passthrough_api_key": "pd-key-456"}):
        assert adapter_with_api_key._get_api_key() == "config-key-123"


async def test_api_key_none_when_neither_set(adapter: FakePassthroughSafetyAdapter) -> None:
    assert adapter._get_api_key() is None


async def test_authorization_header_sent(adapter_with_api_key: FakePassthroughSafetyAdapter) -> None:
    headers = adapter_with_api_key._build_request_headers()
    assert headers["Authorization"] == "Bearer config-key-123"


async def test_empty_api_key_falls_through_to_provider_data(adapter: FakePassthroughSafetyAdapter) -> None:
    config = PassthroughSafetyConfig(base_url="https://safety.example.com/v1", api_key="")
    a = FakePassthroughSafetyAdapter(config, adapter.shield_store)
    _stub_provider_spec(a)

    with provider_data_ctx({"passthrough_api_key": "pd-key-789"}):
        assert a._get_api_key() == "pd-key-789"


# -- malformed downstream response --


async def test_run_shield_null_results_raises(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield()
    downstream = {"id": "modr-null", "model": "text-moderation-latest", "results": None}

    adapter._client.post.return_value = mock_httpx_response(downstream)
    with pytest.raises(RuntimeError, match="malformed response"):
        await adapter.run_shield(
            RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="hello")])
        )


async def test_run_shield_null_categories(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield()
    downstream = {
        "id": "modr-nullcat",
        "model": "text-moderation-latest",
        "results": [{"flagged": True, "categories": None, "category_scores": None}],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    result = await adapter.run_shield(
        RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="bad")])
    )

    assert result.violation is not None
    assert result.violation.metadata["violation_type"] == "unsafe"


async def test_run_moderation_null_results_raises(adapter: FakePassthroughSafetyAdapter) -> None:
    downstream = {"id": "modr-null", "model": "text-moderation-latest", "results": None}

    adapter._client.post.return_value = mock_httpx_response(downstream)
    with pytest.raises(RuntimeError, match="malformed response"):
        await adapter.run_moderation(RunModerationRequest(input="test", model="text-moderation-latest"))


async def test_run_moderation_null_categories(adapter: FakePassthroughSafetyAdapter) -> None:
    downstream = {
        "id": "modr-nullcat",
        "model": "text-moderation-latest",
        "results": [{"flagged": True, "categories": None, "category_scores": None}],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    result = await adapter.run_moderation(RunModerationRequest(input="test", model="text-moderation-latest"))

    assert result.results[0].flagged is True
    assert result.results[0].categories == {}
    assert result.results[0].category_scores == {}


# -- multi-result safety bypass --


async def test_run_shield_flags_later_results(adapter: FakePassthroughSafetyAdapter) -> None:
    """Flagged content in results[1] is caught even when results[0] is safe."""
    adapter.shield_store.get_shield.return_value = _make_shield()
    downstream = {
        "id": "modr-multi",
        "model": "text-moderation-latest",
        "results": [
            {"flagged": False, "categories": {}, "category_scores": {}},
            {"flagged": True, "categories": {"hate": True}, "category_scores": {"hate": 0.95}},
        ],
    }

    adapter._client.post.return_value = mock_httpx_response(downstream)
    result = await adapter.run_shield(
        RunShieldRequest(
            shield_id="test-shield",
            messages=[
                OpenAIUserMessageParam(content="safe"),
                OpenAIUserMessageParam(content="hateful"),
            ],
        )
    )

    assert result.violation is not None
    assert result.violation.metadata["violation_type"] == "hate"


# -- error wrapping --


async def test_run_shield_timeout_wrapped(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield()
    adapter._client.post.side_effect = httpx.ReadTimeout("timed out")

    with pytest.raises(RuntimeError, match="timed out"):
        await adapter.run_shield(
            RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="test")])
        )


async def test_run_shield_connect_error_wrapped(adapter: FakePassthroughSafetyAdapter) -> None:
    adapter.shield_store.get_shield.return_value = _make_shield()
    adapter._client.post.side_effect = httpx.ConnectError("connection refused")

    with pytest.raises(RuntimeError, match="connection failed"):
        await adapter.run_shield(
            RunShieldRequest(shield_id="test-shield", messages=[OpenAIUserMessageParam(content="test")])
        )


async def test_run_moderation_non_json_response(adapter: FakePassthroughSafetyAdapter) -> None:
    resp = mock_httpx_response({})
    resp.json.side_effect = ValueError("No JSON")

    adapter._client.post.return_value = resp
    with pytest.raises(RuntimeError, match="non-JSON response"):
        await adapter.run_moderation(RunModerationRequest(input="test", model="text-moderation-latest"))


async def test_run_moderation_non_dict_json_response(adapter: FakePassthroughSafetyAdapter) -> None:
    resp = mock_httpx_response({})
    resp.json.return_value = ["not", "a", "dict"]

    adapter._client.post.return_value = resp
    with pytest.raises(RuntimeError, match="invalid response"):
        await adapter.run_moderation(RunModerationRequest(input="test", model="text-moderation-latest"))
