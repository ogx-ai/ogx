# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from ogx.providers.inline.responses.builtin.responses.types import AssistantMessageWithReasoning
from ogx.providers.remote.inference.sglang.config import SGLangInferenceAdapterConfig
from ogx.providers.remote.inference.sglang.sglang import SGLangInferenceAdapter
from ogx.providers.utils.inference.systemone import LOCAL_ENGINE_API_KEY
from ogx_api import (
    HealthStatus,
    ModelType,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIUserMessageParam,
    SystemOneRequest,
    SystemOneResponse,
)
from ogx_api.inference import RerankRequest

# Unit tests for the remote SGLang provider. They cover what is specific to
# SGLang's HTTP surface (health probe, rerank wire format, reasoning history);
# API-level behaviour is covered by tests/integration/inference/.
#
# pytest tests/unit/providers/inference/test_remote_sglang.py -v


def _adapter(**config) -> SGLangInferenceAdapter:
    config.setdefault("base_url", "http://mocked.localhost:30000/v1")
    adapter = SGLangInferenceAdapter(config=SGLangInferenceAdapterConfig(**config))
    adapter.__provider_id__ = "sglang"
    return adapter


def _mock_client(mock_client_class, response):
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    mock_client_class.return_value.__aenter__.return_value = client
    return client


async def test_initialize_requires_base_url():
    adapter = SGLangInferenceAdapter(config=SGLangInferenceAdapterConfig())
    with pytest.raises(ValueError, match="SGLANG_URL"):
        await adapter.initialize()


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("http://localhost:30000/v1", "http://localhost:30000"),
        ("http://localhost:30000/v1/", "http://localhost:30000"),
        ("http://localhost:30000", "http://localhost:30000"),
        ("http://localhost:30000/", "http://localhost:30000"),
    ],
)
def test_server_root_strips_version(base_url, expected):
    assert _adapter(base_url=base_url)._get_base_url_without_version() == expected


async def test_health_probes_server_root():
    adapter = _adapter(base_url="http://localhost:30000/v1/")
    with patch("httpx2.AsyncClient") as mock_client_class:
        response = MagicMock()
        response.raise_for_status.return_value = None
        client = _mock_client(mock_client_class, response)

        health = await adapter.health()

    assert health["status"] == HealthStatus.OK
    client.get.assert_called_once_with("http://localhost:30000/health")


async def test_health_reports_connection_errors():
    adapter = _adapter()
    with patch("httpx2.AsyncClient") as mock_client_class:
        client = MagicMock()
        client.get.side_effect = Exception("Connection failed")
        mock_client_class.return_value.__aenter__.return_value = client

        health = await adapter.health()

    assert health["status"] == HealthStatus.ERROR
    assert "Health check failed: Connection failed" in health["message"]


def test_models_are_classified_with_models_dev():
    adapter = _adapter()
    assert adapter.construct_model_from_identifier("text-embedding-3-large").model_type == ModelType.embedding
    assert adapter.construct_model_from_identifier("text-embedding-3-large").provider_id == "sglang"
    assert adapter.construct_model_from_identifier("zai-org/GLM-5.3-Flash").model_type == ModelType.llm


async def test_chat_completion_applies_configured_max_tokens():
    adapter = _adapter(max_tokens=512)
    captured = {}

    async def fake_super(params):
        captured["params"] = params
        return MagicMock()

    with patch.object(
        SGLangInferenceAdapter.__mro__[2], "openai_chat_completion", new=AsyncMock(side_effect=fake_super)
    ):
        await adapter.openai_chat_completion(
            OpenAIChatCompletionRequestWithExtraBody(
                model="m", messages=[OpenAIUserMessageParam(role="user", content="hi")]
            )
        )

    assert captured["params"].max_tokens == 512


async def test_reasoning_history_is_sent_as_reasoning_content():
    """SGLang reads prior-turn thinking from ``reasoning_content``, so the key must survive."""
    adapter = _adapter()
    captured = {}

    async def fake_chat_completion(params):
        captured["messages"] = params.messages

        async def chunks():
            if False:
                yield None

        return chunks()

    adapter.openai_chat_completion = fake_chat_completion
    params = OpenAIChatCompletionRequestWithExtraBody(
        model="m",
        stream=True,
        messages=[
            OpenAIUserMessageParam(role="user", content="hi"),
            AssistantMessageWithReasoning(role="assistant", content="ok", reasoning_content="thinking"),
        ],
    )
    stream = await adapter.openai_chat_completions_with_reasoning(params)
    assert [chunk async for chunk in stream] == []

    assistant = captured["messages"][1]
    assert assistant["reasoning_content"] == "thinking"
    assert "reasoning" not in assistant


async def test_non_streaming_reasoning_is_not_supported():
    adapter = _adapter()
    params = OpenAIChatCompletionRequestWithExtraBody(
        model="m", messages=[OpenAIUserMessageParam(role="user", content="hi")]
    )
    with pytest.raises(NotImplementedError):
        await adapter.openai_chat_completions_with_reasoning(params)


async def test_rerank_uses_sglang_wire_format():
    adapter = _adapter(base_url="https://sglang.example.com/v1", api_token="my-secret-token")
    adapter.get_request_provider_data = MagicMock(return_value=None)

    with patch("httpx2.AsyncClient") as mock_client_class:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [
            {"score": 0.1, "index": 0, "meta_info": {}},
            {"score": 0.9, "index": 1, "document": "doc2"},
        ]
        client = _mock_client(mock_client_class, response)

        result = await adapter.rerank(
            RerankRequest(model="rerank-model", query="test", items=["doc1", "doc2"], max_num_results=2)
        )

    call = client.post.call_args
    assert call.args[0] == "https://sglang.example.com/v1/rerank"
    assert call.kwargs["json"] == {
        "query": "test",
        "documents": ["doc1", "doc2"],
        "top_n": 2,
    }
    assert call.kwargs["headers"]["Authorization"] == "Bearer my-secret-token"
    assert [(d.index, d.relevance_score) for d in result.data] == [(1, 0.9), (0, 0.1)]


async def test_rerank_without_token_sends_no_auth_header():
    adapter = _adapter()
    adapter.get_request_provider_data = MagicMock(return_value=None)

    with patch("httpx2.AsyncClient") as mock_client_class:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [{"score": 0.5, "index": 0}]
        client = _mock_client(mock_client_class, response)

        await adapter.rerank(RerankRequest(model="rerank-model", query="test", items=["doc1"]))

    assert "Authorization" not in client.post.call_args.kwargs["headers"]


@pytest.mark.parametrize("body", [{"results": []}, [{"index": 0}]])
async def test_rerank_rejects_unexpected_responses(body):
    adapter = _adapter()
    adapter.get_request_provider_data = MagicMock(return_value=None)

    with patch("httpx2.AsyncClient") as mock_client_class:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = body
        _mock_client(mock_client_class, response)

        with pytest.raises(RuntimeError):
            await adapter.rerank(RerankRequest(model="rerank-model", query="test", items=["doc1"]))


async def test_rerank_passes_tls_config():
    adapter = _adapter(base_url="https://sglang.example.com/v1", network={"tls": {"verify": False}})
    adapter.get_request_provider_data = MagicMock(return_value=None)

    with patch("httpx2.AsyncClient") as mock_client_class:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [{"score": 0.5, "index": 0}]
        _mock_client(mock_client_class, response)

        await adapter.rerank(RerankRequest(model="rerank-model", query="test", items=["doc1"]))

    assert mock_client_class.call_args.kwargs.get("verify") is False


# ---------------------------------------------------------------------------
# SystemOne (judgment)
# ---------------------------------------------------------------------------


def _judge_request() -> SystemOneRequest:
    return SystemOneRequest(
        model="Cloudflare/clef-flash",
        state="I was charged twice. Please help.",
        questions={"billing": {"type": "noul", "instructions": "Is this about billing?"}},
    )


def _judge_response() -> SystemOneResponse:
    return SystemOneResponse(
        model="Cloudflare/clef-flash",
        answers={"billing": {"type": "noul", "noul": 0.9}},
        usage={"input_tokens": 1, "output_tokens": 0},
    )


async def test_judge_forwards_to_systemone_util():
    """judge() must hit the server root (no /v1) with the local placeholder key and no network config."""
    adapter = _adapter(base_url="http://mocked.localhost:30000/v1/")
    with patch("ogx.providers.remote.inference.sglang.sglang.systemone_judge") as mock_judge:
        mock_judge.return_value = _judge_response()
        await adapter.judge(_judge_request())

    mock_judge.assert_awaited_once()
    args, kwargs = mock_judge.call_args
    assert args[0] == "http://mocked.localhost:30000"
    assert args[1] == LOCAL_ENGINE_API_KEY
    assert kwargs["network"] is None
    # SGLang's x_label_mass extension must be dropped before the strict wire validation.
    assert kwargs["extension_fields"] == frozenset({"x_label_mass"})


async def test_judge_passes_auth_credential_to_systemone_util():
    adapter = _adapter(api_token=SecretStr("top-secret"))
    with patch("ogx.providers.remote.inference.sglang.sglang.systemone_judge") as mock_judge:
        mock_judge.return_value = _judge_response()
        await adapter.judge(_judge_request())

    assert mock_judge.call_args.args[1] == "top-secret"


async def test_judge_passes_network_config_to_systemone_util():
    adapter = _adapter(network={"tls": {"verify": False}})
    with patch("ogx.providers.remote.inference.sglang.sglang.systemone_judge") as mock_judge:
        mock_judge.return_value = _judge_response()
        await adapter.judge(_judge_request())

    assert mock_judge.call_args.kwargs["network"] == adapter.config.network


@pytest.mark.parametrize(
    "identifier",
    [
        "Cloudflare/clef",
        "Cloudflare/clef-flash",
        "Cloudflare/clef-omni",
        "clef-flash",
        "perplexity-ai/pplx-decider-v1-27b",
        "perplexity-ai/pplx-decider-v1.1-27b",
        "/models/Clef-Flash",
        "togethercomputer/Tev1-0.8B-experimental",
        "Tev1-0.8B-experimental",
        "tev1-0.8b",
        "/models/TEV1-0.8B-EXPERIMENTAL",
        "togethercomputer/Tev1-7B-experimental",
        "Tev1",
        "tev1-0.8b-experimental-chat",
    ],
)
def test_construct_model_from_identifier_marks_decision_models(identifier):
    model = _adapter().construct_model_from_identifier(identifier)

    assert model.model_type == ModelType.systemone


@pytest.mark.parametrize(
    "identifier",
    [
        "Qwen/Qwen3-0.6B",
        "zai-org/GLM-5.3-Flash",
        "Cloudflare/clef-2",
        "text-embedding-3-large",
        "tev10-0.8b",
        "my-tev1-0.8b",
    ],
)
def test_construct_model_from_identifier_leaves_other_models_alone(identifier):
    model = _adapter().construct_model_from_identifier(identifier)

    assert model.model_type != ModelType.systemone
