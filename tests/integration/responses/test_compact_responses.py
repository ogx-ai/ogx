# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from ogx_client import OgxClient

from .helpers import skip_if_provider_is_vertexai


@pytest.fixture(autouse=True)
def _skip_compact_tests_for_watsonx(request):
    """Skip compact tests for WatsonX — no recordings available yet."""
    if "text_model_id" in request.fixturenames:
        text_model_id = request.getfixturevalue("text_model_id")
        if text_model_id and text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX compact test recordings not available yet")


class TestCompactResponses:
    """Tests for POST /v1/responses/compact endpoint.

    Note: These tests use the native OpenAI client.responses.compact() method
    for better type safety and IDE support. When dict-style access is needed
    for specific assertions (e.g., checking response structure), we use
    .model_dump() to convert the typed response to a dictionary.

    Benefits of this approach:
    1. Type safety: Direct access to typed fields like result.object, result.usage.input_tokens
    2. IDE support: Full autocomplete and type checking during development
    3. API compatibility: Uses official OpenAI SDK patterns and method signatures
    4. Future-proof: Follows OpenAI SDK evolution and best practices
    """

    @pytest.fixture(autouse=True)
    def _skip_non_openai_client(self, responses_client):
        if isinstance(responses_client, OgxClient):
            pytest.skip("Compact tests require OpenAI client")

    def test_compact_basic_conversation(self, responses_client, text_model_id):
        """Compact a multi-turn conversation with input array."""
        result = responses_client.responses.compact(
            model=text_model_id,
            input=[
                {"role": "user", "content": "Help me plan a Python web app."},
                {"role": "assistant", "content": "I suggest FastAPI with SQLite."},
                {"role": "user", "content": "Add authentication too."},
                {"role": "assistant", "content": "Use OAuth2 with JWT tokens."},
            ],
        )

        # Type-safe access to common fields
        assert result.object == "response.compaction"
        assert result.usage.input_tokens > 0

        # Convert to dict for detailed output inspection
        result_dict = result.model_dump()
        output = result_dict["output"]
        messages = [o for o in output if o.get("type") == "message"]
        compactions = [o for o in output if o.get("type") == "compaction"]
        assert len(messages) == 2  # 2 user messages
        assert all(m["role"] == "user" for m in messages)
        assert len(compactions) == 1
        assert compactions[0]["encrypted_content"]
        assert output[-1]["type"] == "compaction"

    def test_compact_single_message(self, responses_client, text_model_id):
        """Edge case: compact with just one user message."""
        result = responses_client.responses.compact(
            model=text_model_id,
            input=[{"role": "user", "content": "Hello!"}],
        )

        # Type-safe access
        assert result.object == "response.compaction"

        # Convert to dict for output inspection
        result_dict = result.model_dump()
        assert len([o for o in result_dict["output"] if o.get("type") == "message"]) == 1
        assert len([o for o in result_dict["output"] if o.get("type") == "compaction"]) == 1

    def test_compact_with_tool_calls_dropped(self, responses_client, text_model_id):
        """Tool calls and outputs should be dropped from compacted output."""
        result = responses_client.responses.compact(
            model=text_model_id,
            input=[
                {"role": "user", "content": "What's the weather?"},
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "SF"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"temp": 65}',
                },
                {"role": "assistant", "content": "It's 65F in SF."},
                {"role": "user", "content": "Thanks!"},
            ],
        )

        # Convert to dict for output type inspection
        result_dict = result.model_dump()
        output = result_dict["output"]
        types = [o.get("type") for o in output]
        assert "function_call" not in types
        assert "function_call_output" not in types
        assert set(types) == {"message", "compaction"}

    def test_compact_with_previous_response_id(self, responses_client, text_model_id):
        """Compact using previous_response_id to resolve stored history."""
        response = responses_client.responses.create(
            model=text_model_id,
            input="What is the capital of France?",
            store=True,
        )
        result = responses_client.responses.compact(
            model=text_model_id,
            previous_response_id=response.id,
        )

        # Type-safe access
        assert result.object == "response.compaction"

        # Convert to dict for message content inspection
        result_dict = result.model_dump()
        messages = [o for o in result_dict["output"] if o.get("type") == "message"]
        assert any(
            "capital" in m["content"][0]["text"].lower() or "france" in m["content"][0]["text"].lower()
            for m in messages
        )

    def test_compact_roundtrip(self, responses_client, text_model_id):
        """Compact output can be used as input to a new response."""
        compact_result = responses_client.responses.compact(
            model=text_model_id,
            input=[
                {"role": "user", "content": "We're building a book tracker app with FastAPI."},
                {"role": "assistant", "content": "Great choice! Use SQLite for the database."},
                {"role": "user", "content": "What tables do we need?"},
                {"role": "assistant", "content": "Users, Books, and ReadingStatus tables."},
            ],
        )

        # Access output directly from typed response
        followup_input = compact_result.output + [{"role": "user", "content": "What ORM should I use?"}]
        followup = responses_client.responses.create(
            model=text_model_id,
            input=followup_input,
        )
        assert len(followup.output_text) > 0

    def test_compact_input_items_hides_compaction(self, responses_client, text_model_id):
        """input_items should NOT return compaction items."""
        compact_result = responses_client.responses.compact(
            model=text_model_id,
            input=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )

        # Access output directly from typed response
        followup = responses_client.responses.create(
            model=text_model_id,
            input=compact_result.output + [{"role": "user", "content": "How are you?"}],
            store=True,
        )
        items = responses_client.responses.input_items.list(followup.id)
        for item in items.data:
            assert item.type != "compaction"

    def test_compact_chain_through_compaction(self, responses_client, text_model_id):
        """previous_response_id should work directly on a compacted response."""
        compact_result = responses_client.responses.compact(
            model=text_model_id,
            input=[
                {"role": "user", "content": "Remember: the secret word is 'banana'."},
                {"role": "assistant", "content": "Got it, I'll remember the secret word is banana."},
            ],
        )

        # Chain directly through the compacted response ID — no manual concatenation needed
        resp = responses_client.responses.create(
            model=text_model_id,
            input="What was the secret word?",
            previous_response_id=compact_result.id,
        )
        assert "banana" in resp.output_text.lower()

    def test_compact_response_is_retrievable(self, responses_client, text_model_id):
        """Compacted response should be stored and retrievable by ID."""
        compact_result = responses_client.responses.compact(
            model=text_model_id,
            input=[
                {"role": "user", "content": "Hello there!"},
                {"role": "assistant", "content": "Hi! How can I help?"},
            ],
        )

        retrieved = responses_client.responses.retrieve(compact_result.id)
        assert retrieved.id == compact_result.id
        assert retrieved.status == "completed"

    def test_compact_double_compaction(self, responses_client, text_model_id):
        """Compacting an already-compacted conversation should work."""
        c1 = responses_client.responses.compact(
            model=text_model_id,
            input=[
                {"role": "user", "content": "Topic A discussion"},
                {"role": "assistant", "content": "Response about A"},
            ],
        )

        # Access output directly from typed response
        extended = c1.output + [
            {"role": "user", "content": "Topic B discussion"},
            {"role": "assistant", "content": "Response about B"},
        ]

        c2 = responses_client.responses.compact(
            model=text_model_id,
            input=extended,
        )

        # Convert to dict for compaction count inspection
        c2_dict = c2.model_dump()
        compactions = [o for o in c2_dict["output"] if o.get("type") == "compaction"]
        assert len(compactions) == 1

    def test_compact_error_no_input(self, responses_client, text_model_id):
        """Compact with no input and no previous_response_id should error."""
        import openai

        with pytest.raises(openai.BadRequestError):
            responses_client.responses.compact(
                model=text_model_id,
            )


class TestContextManagement:
    """Tests for context_management parameter on responses.create."""

    @pytest.fixture(autouse=True)
    def _skip_non_openai_client(self, responses_client):
        if isinstance(responses_client, OgxClient):
            pytest.skip("Context management tests require OpenAI client")

    @pytest.fixture(autouse=True)
    def _skip_vertexai(self, client_with_models, text_model_id):
        skip_if_provider_is_vertexai(
            client_with_models,
            text_model_id,
            reason="tiktoken does not support vertexai model names for token counting",
        )

    def test_context_management_auto_compacts_large_input(self, responses_client, text_model_id):
        """When input exceeds compact_threshold, context should be auto-compacted."""
        large_input = []
        for i in range(50):
            large_input.append({"role": "user", "content": f"Tell me about topic number {i} in great detail."})
            large_input.append(
                {
                    "role": "assistant",
                    "content": f"Here is a detailed response about topic {i}. " * 20,
                }
            )
        large_input.append({"role": "user", "content": "Summarize what we discussed."})

        response = responses_client.responses.create(
            model=text_model_id,
            input=large_input,
            context_management=[{"type": "compaction", "compact_threshold": 100}],
        )
        assert len(response.output_text) > 0

    def test_context_management_no_compact_below_threshold(self, responses_client, text_model_id):
        """When input is below compact_threshold, no compaction should occur."""
        response = responses_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "Hello!"}],
            context_management=[{"type": "compaction", "compact_threshold": 100000}],
        )
        assert len(response.output_text) > 0

    def test_context_management_none_does_not_compact(self, responses_client, text_model_id):
        """Without context_management, no compaction occurs regardless of input size."""
        response = responses_client.responses.create(
            model=text_model_id,
            input="Hello!",
        )
        assert len(response.output_text) > 0
