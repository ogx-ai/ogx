# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from ogx.providers.remote.inference.vllm.config import VLLMInferenceAdapterConfig
from ogx.providers.remote.inference.vllm.vllm import VLLMInferenceAdapter
from ogx.providers.utils.inference.systemone import LOCAL_ENGINE_API_KEY
from ogx_api import ModelType, SystemOneRequest, SystemOneResponse

# Unit tests for the remote vLLM provider's SystemOne (judgment) support, split out
# from test_remote_vllm.py to stay under the repo's per-file line limit. Mirrors
# test_remote_sglang.py's equivalent tests.
#
# pytest tests/unit/providers/inference/test_remote_vllm_systemone.py -v


def _adapter(**config) -> VLLMInferenceAdapter:
    config.setdefault("base_url", "http://mocked.localhost:8000/v1")
    adapter = VLLMInferenceAdapter(config=VLLMInferenceAdapterConfig(**config))
    adapter.__provider_id__ = "vllm"
    return adapter


def _judge_request() -> SystemOneRequest:
    return SystemOneRequest(
        model="togethercomputer/Tev1-0.8B-experimental",
        state="I was charged twice. Please help.",
        questions={"billing": {"type": "noul", "instructions": "Is this about billing?"}},
    )


def _judge_response() -> SystemOneResponse:
    return SystemOneResponse(
        model="togethercomputer/Tev1-0.8B-experimental",
        answers={"billing": {"type": "noul", "noul": 0.9}},
        usage={"input_tokens": 1, "output_tokens": 0},
    )


async def test_judge_forwards_to_systemone_util():
    """judge() must hit the server root (no /v1) with the local placeholder key and no network config."""
    adapter = _adapter(base_url="http://mocked.localhost:8000/v1/")
    with patch("ogx.providers.remote.inference.vllm.vllm.systemone_judge") as mock_judge:
        mock_judge.return_value = _judge_response()
        await adapter.judge(_judge_request())

    mock_judge.assert_awaited_once()
    args, kwargs = mock_judge.call_args
    assert args[0] == "http://mocked.localhost:8000"
    assert args[1] == LOCAL_ENGINE_API_KEY
    assert kwargs["network"] is None
    # vLLM's own id/object/created/diagnostics fields sit alongside `answers`, not
    # nested inside each answer, so they need top-level stripping.
    assert kwargs["top_level_extension_fields"] == frozenset({"id", "object", "created", "diagnostics"})
    # vLLM's noul answers carry probabilities/confidence too, but choice/score answers
    # need those same field names as required SystemOne fields, so only noul answers
    # may have them stripped.
    assert kwargs["extension_fields_by_type"] == {"noul": frozenset({"probabilities", "confidence"})}


async def test_judge_passes_auth_credential_to_systemone_util():
    adapter = _adapter(api_token=SecretStr("top-secret"))
    with patch("ogx.providers.remote.inference.vllm.vllm.systemone_judge") as mock_judge:
        mock_judge.return_value = _judge_response()
        await adapter.judge(_judge_request())

    assert mock_judge.call_args.args[1] == "top-secret"


async def test_judge_passes_network_config_to_systemone_util():
    adapter = _adapter(network={"tls": {"verify": False}})
    with patch("ogx.providers.remote.inference.vllm.vllm.systemone_judge") as mock_judge:
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
