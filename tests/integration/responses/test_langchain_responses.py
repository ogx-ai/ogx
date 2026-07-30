# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from .helpers import extract_text_content


def test_langchain_basic(responses_client, langchain_chat):
    """Test langchain basic request compatibility with Responses."""

    chat = langchain_chat()

    # Simple question
    messages = [HumanMessage(content="What is the capital of France?")]
    response = chat.invoke(messages)

    # Sanity checks
    assert response is not None
    assert isinstance(response, AIMessage)
    assert response.content is not None
    assert len(response.content) > 0

    # Verify usage is reported
    assert response.usage_metadata is not None, "Response should include usage information"
    assert response.usage_metadata["input_tokens"] > 0, "Input tokens should be greater than 0"
    assert response.usage_metadata["output_tokens"] > 0, "Output tokens should be greater than 0"
    assert (
        response.usage_metadata["total_tokens"]
        == response.usage_metadata["input_tokens"] + response.usage_metadata["output_tokens"]
    ), "Total tokens should equal input + output tokens"

    # Extract content and validate
    content = extract_text_content(response.content)
    assert "paris" in content.lower()

    # Call the Responses API directly and verify consistency with langchain's response
    retrieved_response = responses_client.responses.retrieve(response_id=response.id)
    assert content == retrieved_response.output_text


def test_langchain_chain(responses_client, langchain_chat):
    """Test langchain chaining with Responses"""

    chat = langchain_chat()

    # Create a chain with prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant specializing in {topic}."),
            ("human", "{question}"),
        ]
    )

    chain = prompt | chat

    response = chain.invoke({"topic": "geography", "question": "What are the three largest countries by area?"})

    # Sanity checks
    assert response is not None
    assert isinstance(response, AIMessage)
    assert response.content is not None
    assert len(response.content) > 0

    # Verify usage is reported
    assert response.usage_metadata is not None, "Response should include usage information"
    assert response.usage_metadata["input_tokens"] > 0, "Input tokens should be greater than 0"
    assert response.usage_metadata["output_tokens"] > 0, "Output tokens should be greater than 0"
    assert (
        response.usage_metadata["total_tokens"]
        == response.usage_metadata["input_tokens"] + response.usage_metadata["output_tokens"]
    ), "Total tokens should equal input + output tokens"

    # Extract content and validate
    content = extract_text_content(response.content)
    assert all(country in content.lower() for country in ["russia", "canada"])
    # Response varies depending on model
    assert any(country in content.lower() for country in ["china", "united states"])

    # Call the Responses API directly and verify consistency with langchain's response
    retrieved_response = responses_client.responses.retrieve(response_id=response.id)
    assert content == retrieved_response.output_text


def test_langchain_streaming(responses_client, langchain_chat):
    """Test langchain streaming with Responses"""

    chat = langchain_chat()

    messages = [HumanMessage(content="Count from 1 to 10.")]

    # Collect chunks
    chunks = []
    for chunk in chat.stream(messages):
        chunks.append(chunk)

    assert len(chunks) > 0

    # Verify the Responses API was used by checking the last chunk for response metadata
    last_chunk = chunks[-1]
    assert hasattr(last_chunk, "response_metadata"), (
        f"Last chunk missing response_metadata. Chunk type: {type(last_chunk)}"
    )
    assert "object" in last_chunk.response_metadata, (
        f"Response metadata missing 'object' field. Got: {last_chunk.response_metadata}"
    )
    object_type = last_chunk.response_metadata["object"]
    assert object_type == "response", f"Object type should be 'response' (indicates Responses API). Got: {object_type}"

    # Combine chunks to get full content
    parts = []
    for chunk in chunks:
        if chunk.content:
            text = extract_text_content(chunk.content)
            if text:
                parts.append(text)

    full_content = "".join(parts)
    assert len(full_content) > 0

    # Response should contain all numbers 1-10
    assert all(str(i) in full_content for i in range(1, 11))


def test_langchain_multi_turn(responses_client, langchain_chat):
    """Test langchain multi-turn with Responses

    It seems counter-intuitive to pass the conversation history for multi-turn, but my
    understanding is that langchain doesn't support the Conversations API nor maintains
    any context.  Rather, it needs the history in order to reference prior response ids.
    However, as long as we set use_previous_response_id in the ChatOpenAI() call, it only
    includes the latest input and previous_response_id in the Responses API request.
    """

    chat = langchain_chat(use_previous_response_id=True)

    # Maintain conversation history
    conversation = []

    # Turn 1
    msg1 = HumanMessage(content="What is the Eiffel Tower?")
    conversation.append(msg1)

    response1 = chat.invoke(conversation)
    conversation.append(response1)

    # Sanity checks for turn 1
    assert response1 is not None
    assert isinstance(response1, AIMessage)
    assert response1.content is not None
    assert len(response1.content) > 0

    # Call the Responses API directly and verify consistency with langchain's response
    content = extract_text_content(response1.content)
    retrieved_response = responses_client.responses.retrieve(response_id=response1.id)
    assert content == retrieved_response.output_text

    # Turn 2 - Reference previous context
    msg2 = HumanMessage(content="When was it built?")
    conversation.append(msg2)

    response2 = chat.invoke(conversation)
    conversation.append(response2)

    # Sanity checks for turn 2
    assert response2 is not None
    assert isinstance(response2, AIMessage)
    assert response2.content is not None
    assert len(response2.content) > 0

    # Call the Responses API directly and verify consistency with langchain's response
    content = extract_text_content(response2.content)
    retrieved_response = responses_client.responses.retrieve(response_id=response2.id)
    assert content == retrieved_response.output_text

    # The response should reference construction/building, indicating it understood
    # the context from turn 1
    assert any(word in content.lower() for word in ["1889", "built", "construction", "eighteen"])
