# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import ssl
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from pydantic import SecretStr

from ogx.core.routers.inference import InferenceRouter
from ogx.core.routing_tables.models import ModelsRoutingTable
from ogx.providers.remote.inference.vllm.config import VLLMInferenceAdapterConfig
from ogx.providers.remote.inference.vllm.vllm import VLLMInferenceAdapter
from ogx_api import (
    HealthStatus,
    Model,
    OpenAIChatCompletion,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionResponseMessage,
    OpenAIChoice,
    OpenAICompletion,
    OpenAICompletionChoice,
    OpenAICompletionRequestWithExtraBody,
)

# These are unit test for the remote vllm provider
# implementation. This should only contain tests which are specific to
# the implementation details of those classes. More general
# (API-level) tests should be placed in tests/integration/inference/
#
# How to run this test:
#
# pytest tests/unit/providers/inference/test_remote_vllm.py \
# -v -s --tb=short --disable-warnings


@pytest.fixture(scope="function")
async def vllm_inference_adapter():
    config = VLLMInferenceAdapterConfig(base_url="http://mocked.localhost:12345")
    inference_adapter = VLLMInferenceAdapter(config=config)
    inference_adapter.model_store = AsyncMock()
    await inference_adapter.initialize()
    return inference_adapter


async def test_health_status_success(vllm_inference_adapter):
    """
    Test the health method of VLLM InferenceAdapter when the connection is successful.

    This test verifies that the health method returns a HealthResponse with status OK
    when the /health endpoint responds successfully.
    """
    with patch("httpx.AsyncClient") as mock_client_class:
        # Create mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        # Create mock client instance
        mock_client_instance = MagicMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        # Call the health method
        health_response = await vllm_inference_adapter.health()

        # Verify the response
        assert health_response["status"] == HealthStatus.OK

        # Verify that the health endpoint was called
        mock_client_instance.get.assert_called_once()
        call_args = mock_client_instance.get.call_args[0]
        assert call_args[0].endswith("/health")


async def test_health_status_failure(vllm_inference_adapter):
    """
    Test the health method of VLLM InferenceAdapter when the connection fails.

    This test verifies that the health method returns a HealthResponse with status ERROR
    and an appropriate error message when the connection to the vLLM server fails.
    """
    with patch("httpx.AsyncClient") as mock_client_class:
        # Create mock client instance that raises an exception
        mock_client_instance = MagicMock()
        mock_client_instance.get.side_effect = Exception("Connection failed")
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        # Call the health method
        health_response = await vllm_inference_adapter.health()

        # Verify the response
        assert health_response["status"] == HealthStatus.ERROR
        assert "Health check failed: Connection failed" in health_response["message"]


async def test_health_status_no_static_api_key(vllm_inference_adapter):
    """
    Test the health method of VLLM InferenceAdapter when no static API key is provided.

    This test verifies that the health method returns a HealthResponse with status OK
    when the /health endpoint responds successfully, regardless of API token configuration.
    """
    with patch("httpx.AsyncClient") as mock_client_class:
        # Create mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        # Create mock client instance
        mock_client_instance = MagicMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        # Call the health method
        health_response = await vllm_inference_adapter.health()

        # Verify the response
        assert health_response["status"] == HealthStatus.OK


async def test_openai_chat_completion_is_async(vllm_inference_adapter):
    """
    Verify that openai_chat_completion is async and doesn't block the event loop.

    To do this we mock the underlying inference with a sleep, start multiple
    inference calls in parallel, and ensure the total time taken is less
    than the sum of the individual sleep times.
    """
    sleep_time = 0.5

    async def mock_create(*args, **kwargs):
        await asyncio.sleep(sleep_time)
        return OpenAIChatCompletion(
            id="chatcmpl-abc123",
            created=1,
            model="mock-model",
            choices=[
                OpenAIChoice(
                    message=OpenAIChatCompletionResponseMessage(
                        content="nothing interesting",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
        )

    async def do_inference():
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="mock-model",
            messages=[{"role": "user", "content": "one fish two fish"}],
            stream=False,
        )
        await vllm_inference_adapter.openai_chat_completion(params)

    with patch.object(VLLMInferenceAdapter, "client", new_callable=PropertyMock) as mock_create_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
        mock_create_client.return_value = mock_client

        start_time = time.time()
        await asyncio.gather(do_inference(), do_inference(), do_inference(), do_inference())
        total_time = time.time() - start_time

        assert mock_create_client.call_count == 4  # no cheating
        assert total_time < (sleep_time * 3), f"Total time taken: {total_time}s exceeded expected max"


async def test_vllm_completion_extra_body():
    """
    Test that vLLM-specific guided_choice and prompt_logprobs parameters are correctly forwarded
    via extra_body to the underlying OpenAI client through the InferenceRouter.
    """
    # Set up the vLLM adapter
    config = VLLMInferenceAdapterConfig(base_url="http://mocked.localhost:12345")
    vllm_adapter = VLLMInferenceAdapter(config=config)
    vllm_adapter.__provider_id__ = "vllm"
    await vllm_adapter.initialize()

    # Create a mock model store
    mock_model_store = AsyncMock()
    mock_model = Model(identifier="mock-model", provider_resource_id="mock-model", provider_id="vllm")
    mock_model_store.get_model.return_value = mock_model
    mock_model_store.has_model.return_value = True

    # Create a mock dist_registry
    mock_dist_registry = MagicMock()
    mock_dist_registry.get = AsyncMock(return_value=mock_model)
    mock_dist_registry.set = AsyncMock()

    # Set up the routing table
    routing_table = ModelsRoutingTable(
        impls_by_provider_id={"vllm": vllm_adapter},
        dist_registry=mock_dist_registry,
        policy=[],
    )
    # Inject the model store into the adapter
    vllm_adapter.model_store = routing_table

    # Create the InferenceRouter
    router = InferenceRouter(routing_table=routing_table)

    # Patch the OpenAI client
    with patch.object(VLLMInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_property:
        mock_client = MagicMock()
        mock_client.completions.create = AsyncMock(
            return_value=OpenAICompletion(
                id="cmpl-abc123",
                created=1,
                model="mock-model",
                choices=[
                    OpenAICompletionChoice(
                        text="joy",
                        finish_reason="stop",
                        index=0,
                    )
                ],
            )
        )
        mock_client_property.return_value = mock_client

        # Test with guided_choice and prompt_logprobs as extra fields
        params = OpenAICompletionRequestWithExtraBody(
            model="mock-model",
            prompt="I am feeling happy",
            stream=False,
            guided_choice=["joy", "sadness"],
            prompt_logprobs=5,
        )
        await router.openai_completion(params)

        # Verify that the client was called with extra_body containing both parameters
        mock_client.completions.create.assert_called_once()
        call_kwargs = mock_client.completions.create.call_args.kwargs
        assert "extra_body" in call_kwargs
        assert "guided_choice" in call_kwargs["extra_body"]
        assert call_kwargs["extra_body"]["guided_choice"] == ["joy", "sadness"]
        assert "prompt_logprobs" in call_kwargs["extra_body"]
        assert call_kwargs["extra_body"]["prompt_logprobs"] == 5


async def test_vllm_chat_completion_extra_body():
    """
    Test that vLLM-specific parameters (e.g., chat_template_kwargs) are correctly forwarded
    via extra_body to the underlying OpenAI client through the InferenceRouter for chat completion.
    """
    # Set up the vLLM adapter
    config = VLLMInferenceAdapterConfig(base_url="http://mocked.localhost:12345")
    vllm_adapter = VLLMInferenceAdapter(config=config)
    vllm_adapter.__provider_id__ = "vllm"
    await vllm_adapter.initialize()

    # Create a mock model store
    mock_model_store = AsyncMock()
    mock_model = Model(identifier="mock-model", provider_resource_id="mock-model", provider_id="vllm")
    mock_model_store.get_model.return_value = mock_model
    mock_model_store.has_model.return_value = True

    # Create a mock dist_registry
    mock_dist_registry = MagicMock()
    mock_dist_registry.get = AsyncMock(return_value=mock_model)
    mock_dist_registry.set = AsyncMock()

    # Set up the routing table
    routing_table = ModelsRoutingTable(
        impls_by_provider_id={"vllm": vllm_adapter},
        dist_registry=mock_dist_registry,
        policy=[],
    )
    # Inject the model store into the adapter
    vllm_adapter.model_store = routing_table

    # Create the InferenceRouter
    router = InferenceRouter(routing_table=routing_table)

    # Patch the OpenAI client
    with patch.object(VLLMInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_property:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=OpenAIChatCompletion(
                id="chatcmpl-abc123",
                created=1,
                model="mock-model",
                choices=[
                    OpenAIChoice(
                        message=OpenAIChatCompletionResponseMessage(
                            content="test response",
                        ),
                        finish_reason="stop",
                        index=0,
                    )
                ],
            )
        )
        mock_client_property.return_value = mock_client

        # Test with chat_template_kwargs as extra field
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="mock-model",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            chat_template_kwargs={"thinking": True},
        )
        await router.openai_chat_completion(params)

        # Verify that the client was called with extra_body containing chat_template_kwargs
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "extra_body" in call_kwargs
        assert "chat_template_kwargs" in call_kwargs["extra_body"]
        assert call_kwargs["extra_body"]["chat_template_kwargs"] == {"thinking": True}


class TestHealthTLSConfig:
    """Tests that health() honours TLS/network configuration."""

    async def test_health_uses_shared_ssl_context_by_default(self):
        """Without network config, health() should pass shared_ssl_context as verify."""
        config = VLLMInferenceAdapterConfig(base_url="https://vllm.example.com/v1")
        adapter = VLLMInferenceAdapter(config=config)
        await adapter.initialize()

        kwargs = adapter._build_httpx_client_kwargs()
        assert "verify" in kwargs
        assert isinstance(kwargs["verify"], ssl.SSLContext)

    async def test_health_uses_network_tls_verify_false(self):
        """With network.tls.verify=False, health() should disable TLS verification."""
        config = VLLMInferenceAdapterConfig(
            base_url="https://vllm.example.com/v1",
            network={"tls": {"verify": False}},
        )
        adapter = VLLMInferenceAdapter(config=config)
        await adapter.initialize()

        kwargs = adapter._build_httpx_client_kwargs()
        assert kwargs["verify"] is False

    async def test_health_passes_kwargs_to_httpx(self):
        """health() should pass _build_httpx_client_kwargs() to httpx.AsyncClient."""
        config = VLLMInferenceAdapterConfig(
            base_url="https://vllm.example.com/v1",
            network={"tls": {"verify": False}},
        )
        adapter = VLLMInferenceAdapter(config=config)
        await adapter.initialize()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_client_instance = MagicMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            await adapter.health()

            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs.get("verify") is False

    async def test_legacy_tls_verify_migrates_to_network(self):
        """Legacy tls_verify=False should be migrated to network.tls.verify."""
        config = VLLMInferenceAdapterConfig(
            base_url="https://vllm.example.com/v1",
            tls_verify=False,
        )
        adapter = VLLMInferenceAdapter(config=config)
        await adapter.initialize()

        kwargs = adapter._build_httpx_client_kwargs()
        assert kwargs["verify"] is False


class TestRerankTLSAndAuth:
    """Tests that rerank() honours TLS and auth configuration."""

    async def test_rerank_passes_tls_config(self):
        """rerank() should use _build_httpx_client_kwargs() for TLS."""
        config = VLLMInferenceAdapterConfig(
            base_url="https://vllm.example.com/v1",
            network={"tls": {"verify": False}},
        )
        adapter = VLLMInferenceAdapter(config=config)
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [{"index": 0, "relevance_score": 0.9}],
            }
            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            from ogx_api.inference import RerankRequest

            request = RerankRequest(model="rerank-model", query="test", items=["doc1"])
            await adapter.rerank(request)

            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs.get("verify") is False

    async def test_rerank_sends_auth_header(self):
        """rerank() should include Authorization header when api_token is configured."""
        config = VLLMInferenceAdapterConfig(
            base_url="https://vllm.example.com/v1",
            api_token="my-secret-token",
        )
        adapter = VLLMInferenceAdapter(config=config)
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [{"index": 0, "relevance_score": 0.9}],
            }
            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            from ogx_api.inference import RerankRequest

            request = RerankRequest(model="rerank-model", query="test", items=["doc1"])
            await adapter.rerank(request)

            call_args = mock_client_instance.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert headers.get("Authorization") == "Bearer my-secret-token"

    async def test_rerank_no_auth_header_without_token(self):
        """rerank() should not include Authorization header when no api_token is set."""
        config = VLLMInferenceAdapterConfig(base_url="https://vllm.example.com/v1")
        adapter = VLLMInferenceAdapter(config=config)
        await adapter.initialize()
        adapter.get_request_provider_data = MagicMock(return_value=None)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [{"index": 0, "relevance_score": 0.9}],
            }
            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            from ogx_api.inference import RerankRequest

            request = RerankRequest(model="rerank-model", query="test", items=["doc1"])
            await adapter.rerank(request)

            call_args = mock_client_instance.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert "Authorization" not in headers

    async def test_rerank_uses_provider_data_api_key(self):
        """rerank() should use API key from x-ogx-provider-data header over config."""
        config = VLLMInferenceAdapterConfig(
            base_url="https://vllm.example.com/v1",
            api_token="config-token",
        )
        adapter = VLLMInferenceAdapter(config=config)
        await adapter.initialize()

        adapter.get_request_provider_data = MagicMock(
            return_value=SimpleNamespace(vllm_api_token=SecretStr("provider-data-token"))
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [{"index": 0, "relevance_score": 0.9}],
            }
            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client_instance

            from ogx_api.inference import RerankRequest

            request = RerankRequest(model="rerank-model", query="test", items=["doc1"])
            await adapter.rerank(request)

            call_args = mock_client_instance.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert headers.get("Authorization") == "Bearer provider-data-token"
