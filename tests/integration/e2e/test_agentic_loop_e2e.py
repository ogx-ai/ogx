# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
End-to-end tests for the agentic loop with real vLLM inference and PostgreSQL persistence.

These tests validate that the full tool-calling loop works against real infrastructure:
model emits function call → client provides output → model continues → state persisted in PostgreSQL.
"""

import json
import time

import pytest

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'Paris'",
            },
        },
        "required": ["location"],
        "additionalProperties": False,
    },
}

LOOKUP_TOOL = {
    "type": "function",
    "name": "lookup_employee",
    "description": "Look up an employee by name and return their department and role.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Employee name",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}


def _wait_for_db_row(db, table, id_value, timeout=10):
    """Poll the database until a row appears (async writes may lag slightly)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = db.fetchrow(f"SELECT * FROM {table} WHERE id = $1", id_value)  # noqa: S608
        if row is not None:
            return row
        time.sleep(0.5)
    raise AssertionError(f"Row {id_value} not found in {table} within {timeout}s")


@pytest.mark.integration
@pytest.mark.timeout(180, method="thread")
class TestAgenticLoopE2E:
    """E2E tests for the agentic tool-calling loop with real inference and persistence."""

    def test_single_tool_round_trip(self, openai_client, text_model_id, db):
        """Model calls a tool, client provides output, model responds — verify DB persistence."""
        response1 = openai_client.responses.create(
            model=text_model_id,
            input="What is the weather in Tokyo? YOU MUST USE the get_weather function.",
            tools=[WEATHER_TOOL],
            stream=False,
        )

        function_calls = [o for o in response1.output if o.type == "function_call"]
        assert len(function_calls) >= 1, f"Expected function_call, got types: {[o.type for o in response1.output]}"

        call = function_calls[0]
        assert call.name == "get_weather"
        assert call.status == "completed"
        args = json.loads(call.arguments)
        assert "tokyo" in args.get("location", "").lower() or "Tokyo" in args.get("location", "")

        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "It is sunny and 28 degrees Celsius in Tokyo.",
                },
            ],
            tools=[WEATHER_TOOL],
            previous_response_id=response1.id,
            stream=False,
        )

        assert response2.output[0].type == "message"
        assert len(response2.output_text.strip()) > 0

        row1 = _wait_for_db_row(db, "responses", response1.id)
        resp_obj1 = (
            json.loads(row1["response_object"]) if isinstance(row1["response_object"], str) else row1["response_object"]
        )
        assert resp_obj1["status"] == "completed"
        output_types1 = [o["type"] for o in resp_obj1["output"]]
        assert "function_call" in output_types1

        row2 = _wait_for_db_row(db, "responses", response2.id)
        resp_obj2 = (
            json.loads(row2["response_object"]) if isinstance(row2["response_object"], str) else row2["response_object"]
        )
        assert resp_obj2["status"] == "completed"
        assert row2["previous_response_id"] == response1.id

    def test_tool_round_trip_streaming(self, openai_client, text_model_id, db):
        """Streaming agentic loop — verify events, persistence, and DB state."""
        with openai_client.responses.create(
            model=text_model_id,
            input="What is the weather in Paris? YOU MUST USE the get_weather function.",
            tools=[WEATHER_TOOL],
            stream=True,
        ) as stream:
            events = list(stream)

        created_event = next((e for e in events if e.type == "response.created"), None)
        assert created_event is not None
        response_id = created_event.response.id

        completed_event = next((e for e in events if e.type == "response.completed"), None)
        assert completed_event is not None
        response1 = completed_event.response

        function_calls = [o for o in response1.output if o.type == "function_call"]
        assert len(function_calls) >= 1
        call = function_calls[0]

        retrieved = openai_client.responses.retrieve(response_id=response_id)
        retrieved_calls = [o for o in retrieved.output if o.type == "function_call"]
        assert len(retrieved_calls) >= 1
        assert retrieved_calls[0].call_id == call.call_id

        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "It is rainy and 15 degrees Celsius in Paris.",
                },
            ],
            tools=[WEATHER_TOOL],
            previous_response_id=response1.id,
            stream=False,
        )
        assert len(response2.output_text.strip()) > 0

        row = _wait_for_db_row(db, "responses", response_id)
        resp_obj = (
            json.loads(row["response_object"]) if isinstance(row["response_object"], str) else row["response_object"]
        )
        assert resp_obj["status"] == "completed"

    def test_tool_state_persisted_across_turns(self, openai_client, text_model_id, db):
        """Verify the full tool call chain is persisted in PostgreSQL."""
        response1 = openai_client.responses.create(
            model=text_model_id,
            input="What is the weather in London? YOU MUST USE the get_weather function.",
            tools=[WEATHER_TOOL],
            stream=False,
        )
        function_calls = [o for o in response1.output if o.type == "function_call"]
        assert len(function_calls) >= 1
        call = function_calls[0]

        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "It is foggy and 12 degrees Celsius in London.",
                },
            ],
            tools=[WEATHER_TOOL],
            previous_response_id=response1.id,
            stream=False,
        )

        row1 = _wait_for_db_row(db, "responses", response1.id)
        row2 = _wait_for_db_row(db, "responses", response2.id)

        resp_obj1 = (
            json.loads(row1["response_object"]) if isinstance(row1["response_object"], str) else row1["response_object"]
        )
        resp_obj2 = (
            json.loads(row2["response_object"]) if isinstance(row2["response_object"], str) else row2["response_object"]
        )

        call_ids = [o["call_id"] for o in resp_obj1["output"] if o["type"] == "function_call"]
        assert call.call_id in call_ids

        assert resp_obj2["status"] == "completed"
        assert row2["previous_response_id"] == response1.id
        output_types2 = [o["type"] for o in resp_obj2["output"]]
        assert "message" in output_types2

    def test_multi_tool_conversation_with_persistence(self, openai_client, text_model_id, db):
        """Multi-turn agentic conversation: two tool cycles, all four responses in DB."""
        response1 = openai_client.responses.create(
            model=text_model_id,
            input="What is the weather in Berlin? YOU MUST USE the get_weather function.",
            tools=[WEATHER_TOOL],
            stream=False,
        )
        calls1 = [o for o in response1.output if o.type == "function_call"]
        assert len(calls1) >= 1

        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": calls1[0].call_id,
                    "output": "It is snowing and -2 degrees Celsius in Berlin.",
                },
            ],
            tools=[WEATHER_TOOL],
            previous_response_id=response1.id,
            stream=False,
        )
        assert response2.output[0].type == "message"

        response3 = openai_client.responses.create(
            model=text_model_id,
            input="Now check Madrid. YOU MUST USE the get_weather function.",
            tools=[WEATHER_TOOL],
            previous_response_id=response2.id,
            stream=False,
        )
        calls3 = [o for o in response3.output if o.type == "function_call"]
        assert len(calls3) >= 1

        response4 = openai_client.responses.create(
            model=text_model_id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": calls3[0].call_id,
                    "output": "It is sunny and 22 degrees Celsius in Madrid.",
                },
            ],
            tools=[WEATHER_TOOL],
            previous_response_id=response3.id,
            stream=False,
        )
        assert response4.output[0].type == "message"

        for resp_id in [response1.id, response2.id, response3.id, response4.id]:
            row = _wait_for_db_row(db, "responses", resp_id)
            resp_obj = (
                json.loads(row["response_object"])
                if isinstance(row["response_object"], str)
                else row["response_object"]
            )
            assert resp_obj["status"] == "completed"

        row4 = db.fetchrow("SELECT previous_response_id FROM responses WHERE id = $1", response4.id)
        assert row4["previous_response_id"] == response3.id

    def test_agentic_loop_with_conversation(self, openai_client, text_model_id, db):
        """Agentic loop inside a conversation — verify tool state in conversation items table."""
        conversation = openai_client.conversations.create()

        response1 = openai_client.responses.create(
            model=text_model_id,
            input=[
                {
                    "role": "user",
                    "content": "Look up the employee named Alice. YOU MUST USE the lookup_employee function.",
                }
            ],
            tools=[LOOKUP_TOOL],
            conversation=conversation.id,
            stream=False,
        )
        calls = [o for o in response1.output if o.type == "function_call"]
        assert len(calls) >= 1
        assert calls[0].name == "lookup_employee"

        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": calls[0].call_id,
                    "output": "Alice works in the Engineering department as a Senior Developer.",
                },
            ],
            tools=[LOOKUP_TOOL],
            conversation=conversation.id,
            stream=False,
        )
        assert response2.output[0].type == "message"
        assert "engineering" in response2.output_text.lower() or "senior" in response2.output_text.lower()

        items = openai_client.conversations.items.list(conversation.id)
        assert len(items.data) >= 3

        conv_row = db.fetchrow(
            "SELECT id FROM openai_conversations WHERE id = $1",
            conversation.id,
        )
        assert conv_row is not None, f"Conversation {conversation.id} not found in openai_conversations table"

        conv_items = db.fetch(
            "SELECT id, sort_order FROM conversation_items WHERE conversation_id = $1 ORDER BY sort_order",
            conversation.id,
        )
        assert len(conv_items) >= 3, f"Expected >= 3 conversation items in DB, got {len(conv_items)}"
