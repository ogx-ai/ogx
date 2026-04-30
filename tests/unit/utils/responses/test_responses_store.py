# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from ogx.core.storage.datatypes import ResponsesStoreReference, SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.utils.responses.responses_store import ResponsesStore, _apply_include_filter
from ogx_api import (
    InvalidParameterError,
    OpenAIMessageParam,
    OpenAIResponseInput,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageFileSearchToolCallResults,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageReasoningItem,
    OpenAIResponseOutputMessageReasoningSummary,
    OpenAITokenLogProb,
    OpenAIUserMessageParam,
    Order,
    ResponseInputItemNotFoundError,
    ResponseItemInclude,
    ResponseNotFoundError,
)


def build_store(db_path: str, policy: list | None = None) -> ResponsesStore:
    backend_name = f"sql_responses_{uuid4().hex}"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=db_path)})
    return ResponsesStore(
        ResponsesStoreReference(backend=backend_name, table_name="responses"),
        policy=policy or [],
    )


def create_test_response_object(
    response_id: str, created_timestamp: int, model: str = "test-model"
) -> OpenAIResponseObject:
    """Helper to create a test response object."""
    return OpenAIResponseObject(
        id=response_id,
        created_at=created_timestamp,
        model=model,
        object="response",
        output=[],  # Required field
        status="completed",  # Required field
        store=True,
    )


def create_test_response_input(content: str, input_id: str) -> OpenAIResponseInput:
    """Helper to create a test response input."""
    return OpenAIResponseMessage(
        id=input_id,
        content=content,
        role="user",
        type="message",
    )


def create_test_messages(content: str) -> list[OpenAIMessageParam]:
    """Helper to create test messages for chat completion."""
    return [OpenAIUserMessageParam(content=content)]


def create_test_input_image_message(
    input_id: str = "image-input",
    image_url: str = "https://example.com/image.jpg",
) -> OpenAIResponseInput:
    return OpenAIResponseMessage(
        id=input_id,
        role="user",
        content=[OpenAIResponseInputMessageContentImage(image_url=image_url, detail="high")],
        type="message",
    )


def create_test_output_text_message(
    input_id: str = "output-message",
    text: str = "Paris",
) -> OpenAIResponseInput:
    return OpenAIResponseMessage(
        id=input_id,
        role="assistant",
        content=[
            OpenAIResponseOutputMessageContentOutputText(
                text=text,
                logprobs=[OpenAITokenLogProb(token=text, logprob=-0.1)],
            )
        ],
        type="message",
        status="completed",
    )


def create_test_file_search_call(input_id: str = "file-search-call") -> OpenAIResponseInput:
    return OpenAIResponseOutputMessageFileSearchToolCall(
        id=input_id,
        queries=["capital of France"],
        status="completed",
        results=[
            OpenAIResponseOutputMessageFileSearchToolCallResults(
                attributes={},
                file_id="file-123",
                filename="facts.txt",
                score=0.98,
                text="Paris is the capital of France.",
            )
        ],
    )


def create_test_include_filter_items() -> list[OpenAIResponseInput]:
    return [
        create_test_input_image_message(),
        create_test_output_text_message(),
        create_test_file_search_call(),
        OpenAIResponseOutputMessageFunctionToolCall(
            call_id="call-123",
            name="lookup_weather",
            arguments='{"city":"Paris"}',
            id="function-call-1",
            status="completed",
        ),
        OpenAIResponseOutputMessageReasoningItem(
            id="reasoning-1",
            summary=[OpenAIResponseOutputMessageReasoningSummary(text="Reasoning summary")],
            status="completed",
        ),
    ]


def get_image_content(item: OpenAIResponseInput) -> OpenAIResponseInputMessageContentImage:
    assert isinstance(item, OpenAIResponseMessage)
    assert not isinstance(item.content, str)
    assert isinstance(item.content[0], OpenAIResponseInputMessageContentImage)
    return item.content[0]


def get_output_text_content(item: OpenAIResponseInput) -> OpenAIResponseOutputMessageContentOutputText:
    assert isinstance(item, OpenAIResponseMessage)
    assert not isinstance(item.content, str)
    assert isinstance(item.content[0], OpenAIResponseOutputMessageContentOutputText)
    return item.content[0]


def get_file_search_call(item: OpenAIResponseInput) -> OpenAIResponseOutputMessageFileSearchToolCall:
    assert isinstance(item, OpenAIResponseOutputMessageFileSearchToolCall)
    return item


async def test_responses_store_pagination_basic():
    """Test basic pagination functionality for responses store."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        # Create test data with different timestamps
        base_time = int(time.time())
        test_data = [
            ("zebra-resp", base_time + 1),
            ("apple-resp", base_time + 2),
            ("moon-resp", base_time + 3),
            ("banana-resp", base_time + 4),
            ("car-resp", base_time + 5),
        ]

        # Store test responses
        for response_id, timestamp in test_data:
            response = create_test_response_object(response_id, timestamp)
            input_list = [create_test_response_input(f"Input for {response_id}", f"input-{response_id}")]
            messages = create_test_messages(f"Input for {response_id}")
            await store.store_response_object(response, input_list, messages)

        # Wait for all queued writes to complete
        await store.flush()

        # Test 1: First page with limit=2, descending order (default)
        result = await store.list_responses(limit=2, order=Order.desc)
        assert len(result.data) == 2
        assert result.data[0].id == "car-resp"  # Most recent first
        assert result.data[1].id == "banana-resp"
        assert result.has_more is True
        assert result.last_id == "banana-resp"

        # Test 2: Second page using 'after' parameter
        result2 = await store.list_responses(after="banana-resp", limit=2, order=Order.desc)
        assert len(result2.data) == 2
        assert result2.data[0].id == "moon-resp"
        assert result2.data[1].id == "apple-resp"
        assert result2.has_more is True

        # Test 3: Final page
        result3 = await store.list_responses(after="apple-resp", limit=2, order=Order.desc)
        assert len(result3.data) == 1
        assert result3.data[0].id == "zebra-resp"
        assert result3.has_more is False


async def test_responses_store_pagination_ascending():
    """Test pagination with ascending order."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        # Create test data
        base_time = int(time.time())
        test_data = [
            ("delta-resp", base_time + 1),
            ("charlie-resp", base_time + 2),
            ("alpha-resp", base_time + 3),
        ]

        # Store test responses
        for response_id, timestamp in test_data:
            response = create_test_response_object(response_id, timestamp)
            input_list = [create_test_response_input(f"Input for {response_id}", f"input-{response_id}")]
            messages = create_test_messages(f"Input for {response_id}")
            await store.store_response_object(response, input_list, messages)

        # Wait for all queued writes to complete
        await store.flush()

        # Test ascending order pagination
        result = await store.list_responses(limit=1, order=Order.asc)
        assert len(result.data) == 1
        assert result.data[0].id == "delta-resp"  # Oldest first
        assert result.has_more is True

        # Second page with ascending order
        result2 = await store.list_responses(after="delta-resp", limit=1, order=Order.asc)
        assert len(result2.data) == 1
        assert result2.data[0].id == "charlie-resp"
        assert result2.has_more is True


async def test_responses_store_pagination_with_model_filter():
    """Test pagination combined with model filtering."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        # Create test data with different models
        base_time = int(time.time())
        test_data = [
            ("xyz-resp", base_time + 1, "model-a"),
            ("def-resp", base_time + 2, "model-b"),
            ("pqr-resp", base_time + 3, "model-a"),
            ("abc-resp", base_time + 4, "model-b"),
        ]

        # Store test responses
        for response_id, timestamp, model in test_data:
            response = create_test_response_object(response_id, timestamp, model)
            input_list = [create_test_response_input(f"Input for {response_id}", f"input-{response_id}")]
            messages = create_test_messages(f"Input for {response_id}")
            await store.store_response_object(response, input_list, messages)

        # Wait for all queued writes to complete
        await store.flush()

        # Test pagination with model filter
        result = await store.list_responses(limit=1, model="model-a", order=Order.desc)
        assert len(result.data) == 1
        assert result.data[0].id == "pqr-resp"  # Most recent model-a
        assert result.data[0].model == "model-a"
        assert result.has_more is True

        # Second page with model filter
        result2 = await store.list_responses(after="pqr-resp", limit=1, model="model-a", order=Order.desc)
        assert len(result2.data) == 1
        assert result2.data[0].id == "xyz-resp"
        assert result2.data[0].model == "model-a"
        assert result2.has_more is False


async def test_responses_store_pagination_invalid_after():
    """Test error handling for invalid 'after' parameter."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        # Try to paginate with non-existent ID
        with pytest.raises(ValueError, match="Record with id.*'non-existent' not found in table 'responses'"):
            await store.list_responses(after="non-existent", limit=2)


async def test_responses_store_pagination_no_limit():
    """Test pagination behavior when no limit is specified."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        # Create test data
        base_time = int(time.time())
        test_data = [
            ("omega-resp", base_time + 1),
            ("beta-resp", base_time + 2),
        ]

        # Store test responses
        for response_id, timestamp in test_data:
            response = create_test_response_object(response_id, timestamp)
            input_list = [create_test_response_input(f"Input for {response_id}", f"input-{response_id}")]
            messages = create_test_messages(f"Input for {response_id}")
            await store.store_response_object(response, input_list, messages)

        # Wait for all queued writes to complete
        await store.flush()

        # Test without limit (should use default of 50)
        result = await store.list_responses(order=Order.desc)
        assert len(result.data) == 2
        assert result.data[0].id == "beta-resp"  # Most recent first
        assert result.data[1].id == "omega-resp"
        assert result.has_more is False


async def test_responses_store_get_response_object():
    """Test retrieving a single response object."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        # Store a test response
        response = create_test_response_object("test-resp", int(time.time()))
        input_list = [create_test_response_input("Test input content", "input-test-resp")]
        messages = create_test_messages("Test input content")
        await store.store_response_object(response, input_list, messages)

        # Wait for all queued writes to complete
        await store.flush()

        # Retrieve the response
        retrieved = await store.get_response_object("test-resp")
        assert retrieved.id == "test-resp"
        assert retrieved.model == "test-model"
        assert len(retrieved.input) == 1
        assert retrieved.input[0].content == "Test input content"

        # Test error for non-existent response
        with pytest.raises(ResponseNotFoundError, match="Response 'non-existent' not found"):
            await store.get_response_object("non-existent")


async def test_responses_store_input_items_pagination():
    """Test pagination functionality for input items."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        # Store a test response with many inputs with explicit IDs
        response = create_test_response_object("test-resp", int(time.time()))
        input_list = [
            create_test_response_input("First input", "input-1"),
            create_test_response_input("Second input", "input-2"),
            create_test_response_input("Third input", "input-3"),
            create_test_response_input("Fourth input", "input-4"),
            create_test_response_input("Fifth input", "input-5"),
        ]
        messages = create_test_messages("First input")
        await store.store_response_object(response, input_list, messages)

        # Wait for all queued writes to complete
        await store.flush()

        # Verify all items are stored correctly with explicit IDs
        all_items = await store.list_response_input_items("test-resp", order=Order.desc)
        assert len(all_items.data) == 5

        # In desc order: [Fifth, Fourth, Third, Second, First]
        assert all_items.data[0].content == "Fifth input"
        assert all_items.data[0].id == "input-5"
        assert all_items.data[1].content == "Fourth input"
        assert all_items.data[1].id == "input-4"
        assert all_items.data[2].content == "Third input"
        assert all_items.data[2].id == "input-3"
        assert all_items.data[3].content == "Second input"
        assert all_items.data[3].id == "input-2"
        assert all_items.data[4].content == "First input"
        assert all_items.data[4].id == "input-1"

        # Test basic pagination with after parameter using actual IDs
        result = await store.list_response_input_items("test-resp", limit=2, order=Order.desc)
        assert len(result.data) == 2
        assert result.data[0].content == "Fifth input"  # Most recent first (reversed order)
        assert result.data[1].content == "Fourth input"

        # Test pagination using after with actual ID
        result2 = await store.list_response_input_items("test-resp", after="input-5", limit=2, order=Order.desc)
        assert len(result2.data) == 2
        assert result2.data[0].content == "Fourth input"  # Next item after Fifth
        assert result2.data[1].content == "Third input"

        # Test final page
        result3 = await store.list_response_input_items("test-resp", after="input-3", limit=2, order=Order.desc)
        assert len(result3.data) == 2
        assert result3.data[0].content == "Second input"
        assert result3.data[1].content == "First input"

        # Test ascending order pagination
        result_asc = await store.list_response_input_items("test-resp", limit=2, order=Order.asc)
        assert len(result_asc.data) == 2
        assert result_asc.data[0].content == "First input"  # Oldest first
        assert result_asc.data[1].content == "Second input"

        # Test pagination with ascending order
        result_asc2 = await store.list_response_input_items("test-resp", after="input-1", limit=2, order=Order.asc)
        assert len(result_asc2.data) == 2
        assert result_asc2.data[0].content == "Second input"
        assert result_asc2.data[1].content == "Third input"

        # Test error for non-existent after ID
        with pytest.raises(ResponseInputItemNotFoundError, match="Input item 'non-existent' not found"):
            await store.list_response_input_items("test-resp", after="non-existent")

        # Include should not change the returned items when no gated fields are present
        include_result = await store.list_response_input_items(
            "test-resp",
            include=[ResponseItemInclude.file_search_call_results],
        )
        assert len(include_result.data) == 5
        assert [item.id for item in include_result.data] == [item.id for item in all_items.data]

        # Test error for mutually exclusive parameters
        with pytest.raises(InvalidParameterError, match="Cannot specify both 'before' and 'after' parameters"):
            await store.list_response_input_items("test-resp", before="some-id", after="other-id")


def test_apply_include_filter_no_include_hides_gated_fields_without_mutating_source():
    items = create_test_include_filter_items()

    filtered_items = _apply_include_filter(items, None)

    assert get_image_content(filtered_items[0]).image_url is None
    assert get_output_text_content(filtered_items[1]).logprobs is None
    assert get_file_search_call(filtered_items[2]).results is None
    assert filtered_items[3] == items[3]
    assert filtered_items[4] == items[4]

    # Ensure the source items still retain the stored data.
    assert get_image_content(items[0]).image_url == "https://example.com/image.jpg"
    assert get_output_text_content(items[1]).logprobs is not None
    assert get_file_search_call(items[2]).results is not None


def test_apply_include_filter_preserves_requested_fields():
    items = create_test_include_filter_items()

    filtered_items = _apply_include_filter(
        items,
        [
            ResponseItemInclude.message_input_image_image_url,
            ResponseItemInclude.message_output_text_logprobs,
            ResponseItemInclude.file_search_call_results,
        ],
    )

    assert get_image_content(filtered_items[0]).image_url == "https://example.com/image.jpg"
    assert get_output_text_content(filtered_items[1]).logprobs is not None
    assert get_file_search_call(filtered_items[2]).results is not None


def test_apply_include_filter_noop_values_leave_gated_fields_hidden():
    items = create_test_include_filter_items()

    filtered_items = _apply_include_filter(
        items,
        [
            ResponseItemInclude.web_search_call_action_sources,
            ResponseItemInclude.reasoning_encrypted_content,
            ResponseItemInclude.code_interpreter_call_outputs,
            ResponseItemInclude.computer_call_output_output_image_url,
        ],
    )

    assert get_image_content(filtered_items[0]).image_url is None
    assert get_output_text_content(filtered_items[1]).logprobs is None
    assert get_file_search_call(filtered_items[2]).results is None


def test_apply_include_filter_string_message_content_is_unchanged():
    message = OpenAIResponseMessage(
        id="plain-message",
        role="user",
        content="plain text content",
        type="message",
    )

    filtered_items = _apply_include_filter([message], None)

    assert len(filtered_items) == 1
    assert filtered_items[0] is message
    assert filtered_items[0].content == "plain text content"


async def test_responses_store_input_items_before_pagination():
    """Test before pagination functionality for input items."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        # Store a test response with many inputs with explicit IDs
        response = create_test_response_object("test-resp-before", int(time.time()))
        input_list = [
            create_test_response_input("First input", "before-1"),
            create_test_response_input("Second input", "before-2"),
            create_test_response_input("Third input", "before-3"),
            create_test_response_input("Fourth input", "before-4"),
            create_test_response_input("Fifth input", "before-5"),
        ]
        messages = create_test_messages("First input")
        await store.store_response_object(response, input_list, messages)

        # Wait for all queued writes to complete
        await store.flush()

        # Test before pagination with descending order
        # In desc order: [Fifth, Fourth, Third, Second, First]
        # before="before-3" should return [Fifth, Fourth]
        result = await store.list_response_input_items("test-resp-before", before="before-3", order=Order.desc)
        assert len(result.data) == 2
        assert result.data[0].content == "Fifth input"
        assert result.data[1].content == "Fourth input"

        # Test before pagination with limit
        result2 = await store.list_response_input_items(
            "test-resp-before", before="before-2", limit=3, order=Order.desc
        )
        assert len(result2.data) == 3
        assert result2.data[0].content == "Fifth input"
        assert result2.data[1].content == "Fourth input"
        assert result2.data[2].content == "Third input"

        # Test before pagination with ascending order
        # In asc order: [First, Second, Third, Fourth, Fifth]
        # before="before-4" should return [First, Second, Third]
        result3 = await store.list_response_input_items("test-resp-before", before="before-4", order=Order.asc)
        assert len(result3.data) == 3
        assert result3.data[0].content == "First input"
        assert result3.data[1].content == "Second input"
        assert result3.data[2].content == "Third input"

        # Test before with limit in ascending order
        result4 = await store.list_response_input_items("test-resp-before", before="before-5", limit=2, order=Order.asc)
        assert len(result4.data) == 2
        assert result4.data[0].content == "First input"
        assert result4.data[1].content == "Second input"

        # Test error for non-existent before ID
        with pytest.raises(ResponseInputItemNotFoundError, match="Input item 'non-existent' not found"):
            await store.list_response_input_items("test-resp-before", before="non-existent")


async def test_responses_store_input_items_include_filtering():
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        response = create_test_response_object("test-resp-include", int(time.time()))
        input_list = [
            create_test_input_image_message("image-1"),
            create_test_output_text_message("output-1"),
            create_test_file_search_call("file-search-1"),
        ]
        messages = create_test_messages("Image input")
        await store.store_response_object(response, input_list, messages)
        await store.flush()

        result = await store.list_response_input_items(
            "test-resp-include",
            order=Order.asc,
            include=[ResponseItemInclude.message_input_image_image_url],
        )

        assert len(result.data) == 3
        assert get_image_content(result.data[0]).image_url == "https://example.com/image.jpg"
        assert get_output_text_content(result.data[1]).logprobs is None
        assert get_file_search_call(result.data[2]).results is None

        stored_response = await store.get_response_object("test-resp-include")
        assert get_image_content(stored_response.input[0]).image_url == "https://example.com/image.jpg"
        assert get_output_text_content(stored_response.input[1]).logprobs is not None
        assert get_file_search_call(stored_response.input[2]).results is not None


async def test_responses_store_safety_identifier():
    """Test that safety_identifier field is correctly stored and retrieved."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        # Create test response with safety_identifier
        safety_id = "safety-test-12345"
        response = OpenAIResponseObject(
            id="test-safety-resp",
            created_at=int(time.time()),
            model="test-model",
            object="response",
            output=[],
            status="completed",
            store=True,
            safety_identifier=safety_id,
        )
        input_list = [create_test_response_input("Test input", "input-1")]
        messages = create_test_messages("Test input")

        # Store the response
        await store.store_response_object(response, input_list, messages)
        await store.flush()

        # Retrieve and verify safety_identifier is preserved
        retrieved = await store.get_response_object("test-safety-resp")
        assert retrieved.safety_identifier == safety_id

        # Test list_responses also preserves safety_identifier
        list_result = await store.list_responses(limit=10)
        assert len(list_result.data) == 1
        assert list_result.data[0].safety_identifier == safety_id
