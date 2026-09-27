# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from ogx.providers.inline.responses.builtin.responses.types import AssistantMessageWithReasoning
from ogx.providers.remote.inference.ollama.config import OllamaImplConfig
from ogx.providers.remote.inference.ollama.ollama import OllamaInferenceAdapter
from ogx.providers.utils.inference.openai_compat import prepare_openai_completion_params
from ogx_api import (
    HealthStatus,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIUserMessageParam,
)


async def _empty_stream():
    if False:
        yield None


async def test_openai_chat_completions_with_reasoning_keeps_messages_typed():
    """Ollama should remap reasoning fields without widening messages to raw dicts."""
    adapter = OllamaInferenceAdapter(config=OllamaImplConfig(base_url="http://localhost:11434/v1"))
    adapter.__provider_id__ = "ollama"

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_empty_stream())

    with patch.object(type(adapter), "client", new_callable=PropertyMock, return_value=mock_client):
        with patch("ogx.providers.utils.inference.openai_mixin.localize_image_content") as mock_localize:
            mock_localize.return_value = (b"fake_image_data", "jpeg")

            captured_messages = None

            async def _capture_prepare_params(**kwargs):
                nonlocal captured_messages
                captured_messages = kwargs["messages"]
                return await prepare_openai_completion_params(**kwargs)

            with patch(
                "ogx.providers.utils.inference.openai_mixin.prepare_openai_completion_params",
                new=AsyncMock(side_effect=_capture_prepare_params),
            ):
                result = await adapter.openai_chat_completions_with_reasoning(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="test-model",
                        stream=True,
                        messages=[
                            AssistantMessageWithReasoning(
                                role="assistant",
                                content="Previous answer",
                                reasoning_content="Step 1",
                            ),
                            OpenAIUserMessageParam(
                                role="user",
                                content=[
                                    {"type": "text", "text": "What's in this image?"},
                                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
                                ],
                            ),
                        ],
                    )
                )

    assert result is not None
    mock_localize.assert_called_once_with("http://example.com/image.jpg")

    assert captured_messages is not None
    assert type(captured_messages[0]) is OpenAIAssistantMessageParam
    assert captured_messages[0].model_dump(exclude_none=True)["reasoning"] == "Step 1"
    assert "reasoning_content" not in captured_messages[0].model_dump(exclude_none=True)

    mock_client.chat.completions.create.assert_called_once()
    processed_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert processed_messages[0]["reasoning"] == "Step 1"
    assert "reasoning_content" not in processed_messages[0]
    assert processed_messages[1]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh"


async def test_health_ok():
    """health() probes the unauthenticated GET /api/version endpoint and reports OK on success."""
    adapter = OllamaInferenceAdapter(config=OllamaImplConfig(base_url="http://localhost:11434/v1"))
    adapter.__provider_id__ = "ollama"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_client_instance = MagicMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        health_response = await adapter.health()

    assert health_response["status"] == HealthStatus.OK
    mock_client_instance.get.assert_called_once()
    # The /v1 suffix must be stripped so we hit the Ollama root /api/version endpoint.
    assert mock_client_instance.get.call_args[0][0] == "http://localhost:11434/api/version"


async def test_health_error():
    """health() reports ERROR with a message when the version probe fails."""
    adapter = OllamaInferenceAdapter(config=OllamaImplConfig(base_url="http://localhost:11434/v1"))
    adapter.__provider_id__ = "ollama"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_instance.get = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client_class.return_value.__aenter__.return_value = mock_client_instance

        health_response = await adapter.health()

    assert health_response["status"] == HealthStatus.ERROR
    assert "Connection failed" in health_response["message"]
