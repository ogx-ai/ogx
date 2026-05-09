# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .helpers import extract_text_content


def test_langgraph_basic(responses_client, langchain_chat):
    """Test langgraph basic request compatibility with Responses."""

    chat = langchain_chat()

    # Define state
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add]

    # Define node
    def call_model(state: State):
        response = chat.invoke(state["messages"])
        return {"messages": [response]}

    # Build graph
    workflow = StateGraph(State)
    workflow.add_node("model", call_model)
    workflow.set_entry_point("model")
    workflow.add_edge("model", END)

    graph = workflow.compile()

    # Invoke graph
    result = graph.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})

    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) == 2  # Input + output

    message = result["messages"][1]
    assert isinstance(message, AIMessage)
    assert message.content is not None
    assert len(message.content) > 0

    # Some sanity checks to verify that the Responses API was used
    assert hasattr(message, "response_metadata")
    assert "id" in message.response_metadata, f"Response metadata missing 'id' field.  Got: {message.response_metadata}"

    response_id = message.response_metadata["id"]
    assert response_id.startswith("resp_"), f"Response ID should start with 'resp_'.  Got: {response_id}"

    assert "object" in message.response_metadata, (
        f"Response metadata missing 'object' field. Got:  {message.response_metadata}"
    )
    object_type = message.response_metadata["object"]
    assert object_type == "response", f"Object type should be 'response'.  Got: {object_type}"

    # Verify content
    content = extract_text_content(message.content)
    assert "paris" in content.lower()


def test_langgraph_multi_node(responses_client, langchain_chat):
    """Test langgraph multiple nodes with Responses."""

    chat = langchain_chat()

    # Define state
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add]
        topic: str

    # First node: Ask about a topic
    def ask_question(state: State):
        messages = [
            SystemMessage(content=f"You are an expert on {state['topic']}."),
            HumanMessage(content=f"What is important to know about {state['topic']}?"),
        ]
        response = chat.invoke(messages)
        return {"messages": messages + [response]}

    # Second node: Ask follow-up
    def ask_followup(state: State):
        followup = HumanMessage(content="Can you give me a specific example?")
        response = chat.invoke(state["messages"] + [followup])
        return {"messages": [followup, response]}

    # Build graph
    workflow = StateGraph(State)
    workflow.add_node("ask", ask_question)
    workflow.add_node("followup", ask_followup)
    workflow.set_entry_point("ask")
    workflow.add_edge("ask", "followup")
    workflow.add_edge("followup", END)

    graph = workflow.compile()

    # Invoke graph
    result = graph.invoke({"messages": [], "topic": "Python programming"})

    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) == 5  # System + Human + AI + Human + AI

    # Find AI messages
    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    assert len(ai_messages) >= 2, "Should have at least 2 AI responses"

    for i, ai_message in enumerate(ai_messages, 1):
        assert hasattr(ai_message, "response_metadata")
        assert "id" in ai_message.response_metadata, (
            f"Response {i} metadata missing 'id' field. Got: {ai_message.response_metadata}"
        )

        response_id = ai_message.response_metadata["id"]
        assert response_id.startswith("resp_"), f"Response {i} ID should start with 'resp_'. Got: {response_id}"


def test_langgraph_multi_turn(responses_client, langchain_chat):
    """Test langgraph multi-turn with Responses."""

    chat = langchain_chat(use_previous_response_id=True)

    # Define state
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add]
        turn: int

    # Node: Have conversation
    def converse(state: State):
        turn = state.get("turn", 0)
        messages = state["messages"]

        # Add follow-up question if this is a subsequent turn (turn > 0)
        if turn > 0 and turn - 1 < len(follow_ups):
            followup = follow_ups[turn - 1]
            # Invoke with accumulated messages + new follow-up question
            response = chat.invoke(messages + [followup])
            # Return both the follow-up question and response
            return {"messages": [followup, response], "turn": turn + 1}
        else:
            # First turn: just invoke with existing messages
            response = chat.invoke(messages)
            return {"messages": [response], "turn": turn + 1}

    # Conditional: Continue or end
    def should_continue(state: State):
        turn = state.get("turn", 0)
        # Continue while we have more follow-up questions to ask
        return "continue" if turn < len(follow_ups) + 1 else "end"

    # Follow-up questions to ask on subsequent turns
    follow_ups = [
        HumanMessage(content="When was it built?"),
        HumanMessage(content="How tall is it?"),
    ]

    # Build graph
    workflow = StateGraph(State)
    workflow.add_node("chat", converse)
    workflow.set_entry_point("chat")
    workflow.add_conditional_edges("chat", should_continue, {"continue": "chat", "end": END})

    graph = workflow.compile()

    # Invoke graph with initial question
    result = graph.invoke({"messages": [HumanMessage(content="What is the Eiffel Tower?")], "turn": 0})

    assert result is not None
    assert "messages" in result
    assert result["turn"] == 3

    # Should have: HumanMessage (initial), AIMessage (turn 1),
    #              HumanMessage (follow-up 1), AIMessage (turn 2),
    #              HumanMessage (follow-up 2), AIMessage (turn 3)
    messages = result["messages"]
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]

    assert len(human_messages) == 3, f"Should have 3 human messages, got {len(human_messages)}"
    assert len(ai_messages) == 3, f"Should have 3 AI responses, got {len(ai_messages)}"

    # Verify the first follow-up question was asked
    assert "when was it built" in human_messages[1].content.lower(), (
        f"Second human message should be first follow-up question. Got: {human_messages[1].content}"
    )

    # Verify the second follow-up question was asked
    assert "how tall" in human_messages[2].content.lower(), (
        f"Third human message should be second follow-up question. Got: {human_messages[2].content}"
    )

    # Verify that the Responses API was used
    for i, ai_message in enumerate(ai_messages, 1):
        assert hasattr(ai_message, "response_metadata")
        assert "id" in ai_message.response_metadata, (
            f"Turn {i}: Response metadata missing 'id' field. Got: {ai_message.response_metadata}"
        )

        response_id = ai_message.response_metadata["id"]
        assert response_id.startswith("resp_"), f"Turn {i}: Response ID should start with 'resp_'. Got: {response_id}"

    # Verify the second response references the construction date
    second_response = extract_text_content(ai_messages[1].content).lower()
    assert any(word in second_response for word in ["1889", "built", "construction", "eighteen"]), (
        f"Second response should mention when it was built. Got: {second_response}"
    )

    # Verify the third response references the height
    third_response = extract_text_content(ai_messages[2].content).lower()
    assert any(word in third_response for word in ["300", "330", "324", "meter", "feet", "tall", "height"]), (
        f"Third response should mention the height. Got: {third_response}"
    )
