# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from tests.integration.helpers import assert_text_contains
from tests.integration.responses.fixtures.test_cases import ResponsesTestCase
from tests.integration.responses.streaming_assertions import StreamingValidator

from .conftest import active_search_provider


def _require_backend(request):
    """Skip backend tests when no search backend is configured.

    These tests assert on real search content, so they only run under a
    web-search-* setup (or a local run with a backend API key). Under the
    keyless placeholder setup the placeholder test below covers the fallback.
    """
    if active_search_provider() == "placeholder":
        pytest.skip("No web search backend API key is set; run under a web-search-* setup")


# The questions must be unanswerable from the lane model's parametric
# knowledge: the small local model skips the web_search call and answers
# directly for questions it "knows" (the latest Python version failed this
# way ~1 in 3 times in record runs), while it reliably searches for facts it
# does not know (the expert-count questions called the tool 12/12 without
# any forced tool choice).
web_search_test_cases = [
    pytest.param(
        ResponsesTestCase(
            input="How many experts does the Llama 4 Maverick model have?",
            tools=[{"type": "web_search", "search_context_size": "low"}],
            expected="128",
        ),
        id="llama_experts",
    ),
    pytest.param(
        ResponsesTestCase(
            input="How many experts does the Llama 4 Scout model have?",
            tools=[{"type": "web_search_2025_08_26"}],
            expected="16",
        ),
        id="web_search_2025_08_26_type",
    ),
]


@pytest.mark.parametrize("case", web_search_test_cases)
def test_response_non_streaming_web_search(request, responses_client, text_model_id, case):
    _require_backend(request)
    response = responses_client.responses.create(
        model=text_model_id,
        input=case.input,
        tools=case.tools,
        stream=False,
    )
    assert len(response.output) > 1
    for output in response.output[:-1]:
        assert output.type == "web_search_call"
        assert output.status == "completed"
    assert response.output[-1].type == "message"
    assert response.output[-1].status == "completed"
    assert response.output[-1].role == "assistant"
    assert len(response.output[-1].content) > 0
    assert_text_contains(response.output_text, case.expected)


@pytest.mark.parametrize("case", web_search_test_cases)
def test_response_streaming_web_search(request, responses_client, text_model_id, case):
    """Test streaming behavior with web_search tool."""
    _require_backend(request)
    response = responses_client.responses.create(
        model=text_model_id,
        input=case.input,
        tools=case.tools,
        stream=True,
    )

    chunks = list(response)
    validator = StreamingValidator(chunks)
    validator.assert_basic_event_sequence()
    validator.assert_response_consistency()
    validator.assert_has_tool_calls()
    validator.assert_content_quality(case.expected)
    validator.assert_has_incremental_content()

    final_chunk = chunks[-1]
    assert hasattr(final_chunk, "response")

    assert hasattr(final_chunk.response, "output")
    assert len(final_chunk.response.output) > 0

    web_search_calls = [item for item in final_chunk.response.output if item.type == "web_search_call"]
    assert len(web_search_calls) > 0, "Response should contain web_search_call"
    assert web_search_calls[0].status == "completed"

    messages = [item for item in final_chunk.response.output if item.type == "message"]
    assert len(messages) > 0
    assert messages[0].status == "completed"
    assert messages[0].role == "assistant"


def test_response_multi_turn_streaming_web_search(request, responses_client, text_model_id):
    """Test streaming web_search across multiple turns."""
    _require_backend(request)
    # First turn with web search
    response = responses_client.responses.create(
        model=text_model_id,
        input="How many experts does the Llama 4 Maverick model have?",
        tools=[{"type": "web_search", "search_context_size": "low"}],
        stream=True,
    )

    chunks = list(response)
    validator = StreamingValidator(chunks)

    validator.assert_basic_event_sequence()
    validator.assert_response_consistency()
    validator.assert_has_tool_calls()
    validator.assert_content_quality("128")
    validator.assert_has_incremental_content()

    final_chunk = chunks[-1]
    assert hasattr(final_chunk, "response")
    assert len(final_chunk.response.output) > 0

    web_search_calls = [item for item in final_chunk.response.output if item.type == "web_search_call"]
    assert len(web_search_calls) > 0, "First turn should contain web_search_call"
    assert web_search_calls[0].status == "completed"

    # Second turn.  No tools needed, should use context from first turn.
    response = responses_client.responses.create(
        model=text_model_id,
        input="How many of that model's parameters are active during inference?",
        previous_response_id=final_chunk.response.id,
        stream=True,
    )

    chunks = list(response)
    validator = StreamingValidator(chunks)

    validator.assert_basic_event_sequence()
    validator.assert_response_consistency()
    validator.assert_has_incremental_content()

    final_chunk = chunks[-1]
    assert hasattr(final_chunk, "response")
    assert len(final_chunk.response.output_text) > 0


def test_web_search_placeholder(responses_client, text_model_id):
    """Keyless fallback: with no backend configured, web search resolves to the
    placeholder, which completes with a notice instead of failing.

    Structural assertions only: the model's answer to the placeholder notice is
    nondeterministic in live mode.
    """
    if active_search_provider() != "placeholder":
        pytest.skip("A web search backend is configured; the placeholder fallback is not active")

    response = responses_client.responses.create(
        model=text_model_id,
        input="How many experts does the Llama 4 Maverick model have?",
        tools=[{"type": "web_search"}],
        stream=False,
    )
    assert response.status == "completed"

    web_search_calls = [item for item in response.output if item.type == "web_search_call"]
    assert web_search_calls, "Expected a web_search_call in the response output"
    assert web_search_calls[0].status == "completed"

    messages = [item for item in response.output if item.type == "message"]
    assert messages, "Expected a message output"
    assert messages[-1].status == "completed"
    assert messages[-1].role == "assistant"
