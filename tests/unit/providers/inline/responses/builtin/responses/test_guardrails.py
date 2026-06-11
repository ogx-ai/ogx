# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.providers.inline.responses.builtin.responses.streaming import (
    StreamingResponseOrchestrator,
)
from ogx.providers.inline.responses.builtin.responses.types import ChatCompletionContext
from ogx_api.inference.models import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionChunkWithReasoning,
    OpenAIChoiceDelta,
    OpenAIChunkChoice,
)


@pytest.fixture
def mock_moderation_endpoint():
    return "http://localhost:8080/v1/moderations"


@pytest.fixture
def mock_inference_api():
    return AsyncMock()


@pytest.fixture
def mock_context():
    context = AsyncMock(spec=ChatCompletionContext)
    context.tool_context = AsyncMock()
    context.tool_context.previous_tools = {}
    context.messages = []
    context.tool_choice = None
    context.available_tools = MagicMock(return_value=[])
    return context


# ---------------------------------------------------------------------------
# Reasoning guardrail tests
# ---------------------------------------------------------------------------


async def test_guardrailed_reasoning_validated_through_moderation(
    mock_inference_api, mock_context, mock_moderation_endpoint
):
    """Reasoning content must be included in moderation checks when guardrails are enabled."""
    mock_context.model = "test-model"
    mock_context.temperature = None
    mock_context.top_p = None
    mock_context.frequency_penalty = None

    orchestrator = StreamingResponseOrchestrator(
        inference_api=mock_inference_api,
        ctx=mock_context,
        response_id="resp_reasoning_guardrails",
        created_at=0,
        text=MagicMock(),
        max_infer_iters=1,
        tool_executor=MagicMock(),
        instructions=None,
        moderation_endpoint=mock_moderation_endpoint,
        enable_guardrails=True,
    )

    async def completion_result() -> AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
        chunk = OpenAIChatCompletionChunk(
            id="chatcmpl_reasoning",
            choices=[
                OpenAIChunkChoice(
                    index=0,
                    delta=OpenAIChoiceDelta(content=None, role="assistant"),
                    finish_reason="stop",
                )
            ],
            created=1,
            model="test-model",
            object="chat.completion.chunk",
        )
        yield OpenAIChatCompletionChunkWithReasoning(chunk=chunk, reasoning_content="thinking about the answer...")

    mock_guardrails = AsyncMock(return_value=None)
    with patch(
        "ogx.providers.inline.responses.builtin.responses.streaming.run_guardrails",
        mock_guardrails,
    ):
        events = []
        async for event in orchestrator._process_streaming_chunks(completion_result(), output_messages=[]):
            events.append(event)

        # Verify run_guardrails was called with reasoning content included
        mock_guardrails.assert_called()
        moderation_text = mock_guardrails.call_args[0][1]
        assert "thinking about the answer..." in moderation_text

        # Reasoning events should be emitted (after passing moderation)
        reasoning_events = [e for e in events if hasattr(e, "type") and "reasoning" in e.type]
        assert len(reasoning_events) > 0


async def test_guardrailed_reasoning_streams_before_completion(
    mock_inference_api, mock_context, mock_moderation_endpoint
):
    """Reasoning-only streams should not wait until completion before emitting events."""
    mock_context.model = "test-model"
    mock_context.temperature = None
    mock_context.top_p = None
    mock_context.frequency_penalty = None

    orchestrator = StreamingResponseOrchestrator(
        inference_api=mock_inference_api,
        ctx=mock_context,
        response_id="resp_reasoning_realtime",
        created_at=0,
        text=MagicMock(),
        max_infer_iters=1,
        tool_executor=MagicMock(),
        instructions=None,
        moderation_endpoint=mock_moderation_endpoint,
        enable_guardrails=True,
    )

    gate = asyncio.Event()

    async def completion_result() -> AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
        chunk = OpenAIChatCompletionChunk(
            id="chatcmpl_reasoning",
            choices=[
                OpenAIChunkChoice(
                    index=0,
                    delta=OpenAIChoiceDelta(content=None, role="assistant"),
                    finish_reason=None,
                )
            ],
            created=1,
            model="test-model",
            object="chat.completion.chunk",
        )
        yield OpenAIChatCompletionChunkWithReasoning(chunk=chunk, reasoning_content="thinking...")

        await gate.wait()

    mock_guardrails = AsyncMock(return_value=None)
    with patch(
        "ogx.providers.inline.responses.builtin.responses.streaming.run_guardrails",
        mock_guardrails,
    ):
        stream = orchestrator._process_streaming_chunks(completion_result(), output_messages=[])

        first_event = await asyncio.wait_for(anext(stream), timeout=0.5)
        assert first_event.type in {"response.content_part.added", "response.reasoning_text.delta"}
        assert "thinking..." in mock_guardrails.call_args[0][1]

        gate.set()
        async for _ in stream:
            pass


async def test_guardrailed_reasoning_blocked_on_violation(mock_inference_api, mock_context, mock_moderation_endpoint):
    """When moderation flags reasoning content, a refusal should be emitted instead."""
    mock_context.model = "test-model"
    mock_context.temperature = None
    mock_context.top_p = None
    mock_context.frequency_penalty = None

    orchestrator = StreamingResponseOrchestrator(
        inference_api=mock_inference_api,
        ctx=mock_context,
        response_id="resp_reasoning_blocked",
        created_at=0,
        text=MagicMock(),
        max_infer_iters=1,
        tool_executor=MagicMock(),
        instructions=None,
        moderation_endpoint=mock_moderation_endpoint,
        enable_guardrails=True,
    )

    async def completion_result() -> AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
        chunk = OpenAIChatCompletionChunk(
            id="chatcmpl_reasoning",
            choices=[
                OpenAIChunkChoice(
                    index=0,
                    delta=OpenAIChoiceDelta(content=None, role="assistant"),
                    finish_reason="stop",
                )
            ],
            created=1,
            model="test-model",
            object="chat.completion.chunk",
        )
        yield OpenAIChatCompletionChunkWithReasoning(chunk=chunk, reasoning_content="harmful reasoning content")

    mock_guardrails = AsyncMock(return_value="Content blocked by safety guardrails")
    with patch(
        "ogx.providers.inline.responses.builtin.responses.streaming.run_guardrails",
        mock_guardrails,
    ):
        events = []
        async for event in orchestrator._process_streaming_chunks(completion_result(), output_messages=[]):
            events.append(event)

        # Should get a refusal response, no reasoning events
        reasoning_events = [e for e in events if hasattr(e, "type") and "reasoning" in e.type]
        assert len(reasoning_events) == 0
        # The refusal response should be present
        refusal_events = [e for e in events if hasattr(e, "type") and e.type == "response.completed"]
        assert len(refusal_events) == 1
        assert orchestrator.violation_detected is True


# ---------------------------------------------------------------------------
# run_guardrails fail-closed tests
# ---------------------------------------------------------------------------


def _mock_httpx_response(status_code: int = 200, json_data: dict | list | None = None, text: str | None = None):
    """Build a mock httpx.Response for run_guardrails tests."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    if status_code >= 400:
        import httpx

        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=MagicMock(), response=mock_resp)
    else:
        mock_resp.raise_for_status = MagicMock()
    if json_data is not None:
        mock_resp.json.return_value = json_data
    elif text is not None:
        mock_resp.json.side_effect = ValueError("not json")
    return mock_resp


class TestRunGuardrailsFailClosed:
    """run_guardrails must block content on any moderation service error."""

    async def test_http_error_blocks_content(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(status_code=500)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is not None
        assert "Failed to validate content" in result

    async def test_timeout_blocks_content(self):
        import httpx

        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is not None
        assert "Failed to validate content" in result

    async def test_invalid_url_blocks_content(self):
        import httpx

        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.InvalidURL("bad URL"))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://[::1", "hello world")
        assert result is not None
        assert "Failed to validate content" in result

    async def test_invalid_json_blocks_content(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(status_code=200, text="not json at all")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is not None
        assert "Failed to validate content" in result

    async def test_missing_results_field_blocks_content(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(json_data={"id": "modr-123", "model": "text-moderation"})
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is not None
        assert "unexpected format" in result

    async def test_non_list_results_blocks_content(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(json_data={"results": "not a list"})
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is not None
        assert "unexpected format" in result

    async def test_malformed_result_entry_blocks_content(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(json_data={"results": ["not a dict"]})
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is not None
        assert "unexpected format" in result

    async def test_empty_results_blocks_content(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(json_data={"results": []})
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is not None
        assert "unexpected format" in result

    async def test_missing_flagged_field_blocks_content(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(json_data={"results": [{"categories": {"violence": True}}]})
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is not None
        assert "unexpected format" in result

    async def test_malformed_categories_blocks_content(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(json_data={"results": [{"flagged": False, "categories": "bad"}]})
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is not None
        assert "unexpected format" in result

    async def test_flagged_malformed_categories_blocks_content(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(json_data={"results": [{"flagged": True, "categories": None}]})
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is not None
        assert "unexpected format" in result

    async def test_clean_content_returns_none(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(json_data={"results": [{"flagged": False, "categories": {}}]})
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is None

    async def test_flagged_content_returns_message(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(
            json_data={"results": [{"flagged": True, "categories": {"violence": True, "self-harm": False}}]}
        )
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails("http://mod.test/v1/moderations", "hello world")
        assert result is not None
        assert "Content blocked" in result
        assert "violence" in result

    async def test_empty_messages_returns_none(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        result = await run_guardrails("http://mod.test/v1/moderations", "")
        assert result is None

    async def test_headers_are_sent(self):
        from ogx.providers.inline.responses.builtin.responses.utils import run_guardrails

        mock_resp = _mock_httpx_response(json_data={"results": [{"flagged": False, "categories": {}}]})
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            import httpx

            mp.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)
            result = await run_guardrails(
                "http://mod.test/v1/moderations",
                "hello",
                headers={"Authorization": "Bearer sk-test123"},
            )
        assert result is None
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs.get("headers") == {"Authorization": "Bearer sk-test123"}


# ---------------------------------------------------------------------------
# Moderation headers config tests
# ---------------------------------------------------------------------------


class TestModerationHeadersConfig:
    """moderation_headers config field is accepted and threaded correctly."""

    def test_default_is_none(self):
        from ogx.providers.inline.responses.builtin.config import BuiltinResponsesImplConfig

        config = BuiltinResponsesImplConfig(
            persistence={"responses": {"backend": "sql_default", "table_name": "responses"}}
        )
        assert config.moderation_headers is None

    def test_accepts_header_dict(self):
        from ogx.providers.inline.responses.builtin.config import BuiltinResponsesImplConfig

        config = BuiltinResponsesImplConfig(
            persistence={"responses": {"backend": "sql_default", "table_name": "responses"}},
            moderation_endpoint="http://mod.test/v1/moderations",
            moderation_headers={"Authorization": "Bearer sk-test"},
        )
        assert config.moderation_headers == {"Authorization": "Bearer sk-test"}
        assert config.moderation_endpoint == "http://mod.test/v1/moderations"
