# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from ogx_api import (
    GetPromptRequest,
    InvalidParameterError,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIFile,
    OpenAIFileObject,
    OpenAISystemMessageParam,
    Prompt,
)
from ogx_api.inference import (
    OpenAIUserMessageParam,
)
from ogx_api.openai_responses import (
    OpenAIResponseInputMessageContentFile,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponsePrompt,
)
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


async def test_create_openai_response_with_prompt(openai_responses_impl, mock_inference_api, mock_prompts_api):
    """Test creating an OpenAI response with a prompt."""
    input_text = "What is the capital of Ireland?"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="You are a helpful {{ area_name }} assistant at {{ company_name }}. Always provide accurate information.",
        prompt_id=prompt_id,
        version=1,
        variables=["area_name", "company_name"],
        is_default=True,
    )

    openai_response_prompt = OpenAIResponsePrompt(
        id=prompt_id,
        version="1",
        variables={
            "area_name": OpenAIResponseInputMessageContentText(text="geography"),
            "company_name": OpenAIResponseInputMessageContentText(text="Dummy Company"),
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        prompt=openai_response_prompt,
    )

    mock_prompts_api.get_prompt.assert_called_with(GetPromptRequest(prompt_id=prompt_id, version=1))
    mock_inference_api.openai_chat_completion.assert_called()
    call_args = mock_inference_api.openai_chat_completion.call_args
    sent_messages = call_args.args[0].messages
    assert len(sent_messages) == 2

    system_messages = [msg for msg in sent_messages if msg.role == "system"]
    assert len(system_messages) == 1
    assert (
        system_messages[0].content
        == "You are a helpful geography assistant at Dummy Company. Always provide accurate information."
    )

    user_messages = [msg for msg in sent_messages if msg.role == "user"]
    assert len(user_messages) == 1
    assert user_messages[0].content == input_text

    assert result.model == model
    assert result.status == "completed"
    assert isinstance(result.prompt, OpenAIResponsePrompt)
    assert result.prompt.id == prompt_id
    assert result.prompt.variables == openai_response_prompt.variables
    assert result.prompt.version == "1"


async def test_prepend_prompt_successful_without_variables(openai_responses_impl, mock_prompts_api, mock_inference_api):
    """Test prepend_prompt function without variables."""
    input_text = "What is the capital of Ireland?"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="You are a helpful assistant. Always provide accurate information.",
        prompt_id=prompt_id,
        version=1,
        variables=[],
        is_default=True,
    )

    openai_response_prompt = OpenAIResponsePrompt(id=prompt_id, version="1")

    mock_prompts_api.get_prompt.return_value = prompt
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        prompt=openai_response_prompt,
    )

    mock_prompts_api.get_prompt.assert_called_with(GetPromptRequest(prompt_id=prompt_id, version=1))
    mock_inference_api.openai_chat_completion.assert_called()
    call_args = mock_inference_api.openai_chat_completion.call_args
    sent_messages = call_args.args[0].messages
    assert len(sent_messages) == 2
    system_messages = [msg for msg in sent_messages if msg.role == "system"]
    assert system_messages[0].content == "You are a helpful assistant. Always provide accurate information."


async def test_prepend_prompt_invalid_variable(openai_responses_impl, mock_prompts_api):
    """Test error handling in prepend_prompt function when prompt parameters contain invalid variables."""
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="You are a {{ role }} assistant.",
        prompt_id=prompt_id,
        version=1,
        variables=["role"],  # Only "role" is valid
        is_default=True,
    )

    openai_response_prompt = OpenAIResponsePrompt(
        id=prompt_id,
        version="1",
        variables={
            "role": OpenAIResponseInputMessageContentText(text="helpful"),
            "company": OpenAIResponseInputMessageContentText(
                text="Dummy Company"
            ),  # company is not in prompt.variables
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Test prompt")]

    # Execute - should raise InvalidParameterError for invalid variable
    with pytest.raises(InvalidParameterError) as exc_info:
        await openai_responses_impl._prepend_prompt(messages, openai_response_prompt)
    assert "Invalid value for 'prompt.variables': company" in str(exc_info.value)
    assert f"Variable not defined in prompt '{prompt_id}'" in str(exc_info.value)

    # Verify
    mock_prompts_api.get_prompt.assert_called_once_with(GetPromptRequest(prompt_id=prompt_id, version=1))


async def test_prepend_prompt_not_found(openai_responses_impl, mock_prompts_api):
    """Test prepend_prompt function when prompt is not found."""
    prompt_id = "pmpt_nonexistent"
    openai_response_prompt = OpenAIResponsePrompt(id=prompt_id, version="1")

    mock_prompts_api.get_prompt.return_value = None  # Prompt not found

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Test prompt")]
    initial_length = len(messages)

    # Execute
    result = await openai_responses_impl._prepend_prompt(messages, openai_response_prompt)

    # Verify
    mock_prompts_api.get_prompt.assert_called_once_with(GetPromptRequest(prompt_id=prompt_id, version=1))

    # Should return None when prompt not found
    assert result is None

    # Messages should not be modified
    assert len(messages) == initial_length
    assert messages[0].content == "Test prompt"


async def test_prepend_prompt_variable_substitution(openai_responses_impl, mock_prompts_api):
    """Test complex variable substitution with multiple occurrences and special characters in prepend_prompt function."""
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"

    # Support all whitespace variations: {{name}}, {{ name }}, {{ name}}, {{name }}, etc.
    prompt = Prompt(
        prompt="Hello {{name}}! You are working at {{ company}}. Your role is {{role}} at {{company}}. Remember, {{ name }}, to be {{ tone }}.",
        prompt_id=prompt_id,
        version=1,
        variables=["name", "company", "role", "tone"],
        is_default=True,
    )

    openai_response_prompt = OpenAIResponsePrompt(
        id=prompt_id,
        version="1",
        variables={
            "name": OpenAIResponseInputMessageContentText(text="Alice"),
            "company": OpenAIResponseInputMessageContentText(text="Dummy Company"),
            "role": OpenAIResponseInputMessageContentText(text="AI Assistant"),
            "tone": OpenAIResponseInputMessageContentText(text="professional"),
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Test")]

    # Execute
    await openai_responses_impl._prepend_prompt(messages, openai_response_prompt)

    # Verify
    assert len(messages) == 2
    assert isinstance(messages[0], OpenAISystemMessageParam)
    expected_content = "Hello Alice! You are working at Dummy Company. Your role is AI Assistant at Dummy Company. Remember, Alice, to be professional."
    assert messages[0].content == expected_content


async def test_prepend_prompt_with_image_variable(openai_responses_impl, mock_prompts_api, mock_files_api):
    """Test prepend_prompt with image variable - should create placeholder in system message and append image as separate user message."""
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Analyze this {{product_image}} and describe what you see.",
        prompt_id=prompt_id,
        version=1,
        variables=["product_image"],
        is_default=True,
    )

    # Mock file content and file metadata
    mock_file_content = b"fake_image_data"
    mock_files_api.openai_retrieve_file_content.return_value = type("obj", (object,), {"body": mock_file_content})()
    mock_files_api.openai_retrieve_file.return_value = OpenAIFileObject(
        object="file",
        id="file-abc123",
        bytes=len(mock_file_content),
        created_at=1234567890,
        expires_at=1234567890,
        filename="product.jpg",
        purpose="assistants",
        status="processed",
        status_details="",
    )

    openai_response_prompt = OpenAIResponsePrompt(
        id=prompt_id,
        version="1",
        variables={
            "product_image": OpenAIResponseInputMessageContentImage(
                file_id="file-abc123",
                detail="high",
            )
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="What do you think?")]

    # Execute
    await openai_responses_impl._prepend_prompt(messages, openai_response_prompt)

    assert len(messages) == 3

    # Check system message has placeholder
    assert isinstance(messages[0], OpenAISystemMessageParam)
    assert messages[0].content == "Analyze this [Image: product_image] and describe what you see."

    # Check original user message is still there
    assert isinstance(messages[1], OpenAIUserMessageParam)
    assert messages[1].content == "What do you think?"

    # Check new user message with image is appended
    assert isinstance(messages[2], OpenAIUserMessageParam)
    assert isinstance(messages[2].content, list)
    assert len(messages[2].content) == 1

    # Should be image with data URL
    assert isinstance(messages[2].content[0], OpenAIChatCompletionContentPartImageParam)
    assert messages[2].content[0].image_url.url.startswith("data:image/")
    assert messages[2].content[0].image_url.detail == "high"


async def test_prepend_prompt_with_file_variable(openai_responses_impl, mock_prompts_api, mock_files_api):
    """Test prepend_prompt with file variable - should create placeholder in system message and append file as separate user message."""
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Review the document {{contract_file}} and summarize key points.",
        prompt_id=prompt_id,
        version=1,
        variables=["contract_file"],
        is_default=True,
    )

    # Mock file retrieval
    mock_file_content = b"fake_pdf_content"
    mock_files_api.openai_retrieve_file_content.return_value = type("obj", (object,), {"body": mock_file_content})()
    mock_files_api.openai_retrieve_file.return_value = OpenAIFileObject(
        object="file",
        id="file-contract-789",
        bytes=len(mock_file_content),
        created_at=1234567890,
        expires_at=1234567890,
        filename="contract.pdf",
        purpose="assistants",
        status="processed",
        status_details="",
    )

    openai_response_prompt = OpenAIResponsePrompt(
        id=prompt_id,
        version="1",
        variables={
            "contract_file": OpenAIResponseInputMessageContentFile(
                file_id="file-contract-789",
                filename="contract.pdf",
            )
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Please review this.")]

    # Execute
    await openai_responses_impl._prepend_prompt(messages, openai_response_prompt)

    assert len(messages) == 3

    # Check system message has placeholder
    assert isinstance(messages[0], OpenAISystemMessageParam)
    assert messages[0].content == "Review the document [File: contract_file] and summarize key points."

    # Check original user message is still there
    assert isinstance(messages[1], OpenAIUserMessageParam)
    assert messages[1].content == "Please review this."

    # Check new user message with file is appended
    assert isinstance(messages[2], OpenAIUserMessageParam)
    assert isinstance(messages[2].content, list)
    assert len(messages[2].content) == 1

    # First part should be file with data URL
    assert isinstance(messages[2].content[0], OpenAIFile)
    assert messages[2].content[0].file.file_data.startswith("data:application/pdf;base64,")
    assert messages[2].content[0].file.filename == "contract.pdf"
    assert messages[2].content[0].file.file_id is None


async def test_prepend_prompt_with_mixed_variables(openai_responses_impl, mock_prompts_api, mock_files_api):
    """Test prepend_prompt with text, image, and file variables mixed together."""
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Hello {{name}}! Analyze {{photo}} and review {{document}}. Provide insights for {{company}}.",
        prompt_id=prompt_id,
        version=1,
        variables=["name", "photo", "document", "company"],
        is_default=True,
    )

    # Mock file retrieval for image and file
    mock_image_content = b"fake_image_data"
    mock_file_content = b"fake_doc_content"

    async def mock_retrieve_file_content(request):
        file_id = request.file_id
        if file_id == "file-photo-123":
            return type("obj", (object,), {"body": mock_image_content})()
        elif file_id == "file-doc-456":
            return type("obj", (object,), {"body": mock_file_content})()

    mock_files_api.openai_retrieve_file_content.side_effect = mock_retrieve_file_content

    def mock_retrieve_file(request):
        file_id = request.file_id
        if file_id == "file-photo-123":
            return OpenAIFileObject(
                object="file",
                id="file-photo-123",
                bytes=len(mock_image_content),
                created_at=1234567890,
                expires_at=1234567890,
                filename="photo.jpg",
                purpose="assistants",
                status="processed",
                status_details="",
            )
        elif file_id == "file-doc-456":
            return OpenAIFileObject(
                object="file",
                id="file-doc-456",
                bytes=len(mock_file_content),
                created_at=1234567890,
                expires_at=1234567890,
                filename="doc.pdf",
                purpose="assistants",
                status="processed",
                status_details="",
            )

    mock_files_api.openai_retrieve_file.side_effect = mock_retrieve_file

    openai_response_prompt = OpenAIResponsePrompt(
        id=prompt_id,
        version="1",
        variables={
            "name": OpenAIResponseInputMessageContentText(text="Alice"),
            "photo": OpenAIResponseInputMessageContentImage(file_id="file-photo-123", detail="auto"),
            "document": OpenAIResponseInputMessageContentFile(file_id="file-doc-456", filename="doc.pdf"),
            "company": OpenAIResponseInputMessageContentText(text="Acme Corp"),
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="Here's my question.")]

    # Execute
    await openai_responses_impl._prepend_prompt(messages, openai_response_prompt)

    assert len(messages) == 3

    # Check system message has text and placeholders
    assert isinstance(messages[0], OpenAISystemMessageParam)
    expected_system = "Hello Alice! Analyze [Image: photo] and review [File: document]. Provide insights for Acme Corp."
    assert messages[0].content == expected_system

    # Check original user message is still there
    assert isinstance(messages[1], OpenAIUserMessageParam)
    assert messages[1].content == "Here's my question."

    # Check new user message with media is appended (2 media items)
    assert isinstance(messages[2], OpenAIUserMessageParam)
    assert isinstance(messages[2].content, list)
    assert len(messages[2].content) == 2

    # First part should be image with data URL
    assert isinstance(messages[2].content[0], OpenAIChatCompletionContentPartImageParam)
    assert messages[2].content[0].image_url.url.startswith("data:image/")

    # Second part should be file with data URL
    assert isinstance(messages[2].content[1], OpenAIFile)
    assert messages[2].content[1].file.file_data.startswith("data:application/pdf;base64,")
    assert messages[2].content[1].file.filename == "doc.pdf"
    assert messages[2].content[1].file.file_id is None


async def test_prepend_prompt_with_image_using_image_url(openai_responses_impl, mock_prompts_api):
    """Test prepend_prompt with image variable using image_url instead of file_id."""
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Describe {{screenshot}}.",
        prompt_id=prompt_id,
        version=1,
        variables=["screenshot"],
        is_default=True,
    )

    openai_response_prompt = OpenAIResponsePrompt(
        id=prompt_id,
        version="1",
        variables={
            "screenshot": OpenAIResponseInputMessageContentImage(
                image_url="https://example.com/screenshot.png",
                detail="low",
            )
        },
    )

    mock_prompts_api.get_prompt.return_value = prompt

    # Initial messages
    messages = [OpenAIUserMessageParam(content="What is this?")]

    # Execute
    await openai_responses_impl._prepend_prompt(messages, openai_response_prompt)

    assert len(messages) == 3

    # Check system message has placeholder
    assert isinstance(messages[0], OpenAISystemMessageParam)
    assert messages[0].content == "Describe [Image: screenshot]."

    # Check original user message is still there
    assert isinstance(messages[1], OpenAIUserMessageParam)
    assert messages[1].content == "What is this?"

    # Check new user message with image is appended
    assert isinstance(messages[2], OpenAIUserMessageParam)
    assert isinstance(messages[2].content, list)

    # Image should use the provided URL
    assert isinstance(messages[2].content[0], OpenAIChatCompletionContentPartImageParam)
    assert messages[2].content[0].image_url.url == "https://example.com/screenshot.png"
    assert messages[2].content[0].image_url.detail == "low"


async def test_prepend_prompt_image_variable_missing_required_fields(openai_responses_impl, mock_prompts_api):
    """Test prepend_prompt with image variable that has neither file_id nor image_url - should raise error."""
    prompt_id = "pmpt_1234567890abcdef1234567890abcdef1234567890abcdef"
    prompt = Prompt(
        prompt="Analyze {{bad_image}}.",
        prompt_id=prompt_id,
        version=1,
        variables=["bad_image"],
        is_default=True,
    )

    # Create image content with neither file_id nor image_url
    openai_response_prompt = OpenAIResponsePrompt(
        id=prompt_id,
        version="1",
        variables={"bad_image": OpenAIResponseInputMessageContentImage()},  # No file_id or image_url
    )

    mock_prompts_api.get_prompt.return_value = prompt
    messages = [OpenAIUserMessageParam(content="Test")]

    # Execute - should raise ValueError
    with pytest.raises(ValueError, match="Image content must have either 'image_url' or 'file_id'"):
        await openai_responses_impl._prepend_prompt(messages, openai_response_prompt)
