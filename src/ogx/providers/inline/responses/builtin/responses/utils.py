# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import mimetypes
import re
import uuid
from collections.abc import AsyncIterator, Sequence

from ogx.log import get_logger
from ogx.providers.inline.responses.builtin.responses.types import AssistantMessageWithReasoning
from ogx_api import (
    Files,
    Inference,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
    OpenAIDeveloperMessageParam,
    OpenAIFile,
    OpenAIFileFile,
    OpenAIImageURL,
    OpenAIJSONSchema,
    OpenAIMessageParam,
    OpenAIResponseAnnotationFileCitation,
    OpenAIResponseCompaction,
    OpenAIResponseFormatJSONObject,
    OpenAIResponseFormatJSONSchema,
    OpenAIResponseFormatParam,
    OpenAIResponseFormatText,
    OpenAIResponseInput,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseInputMessageContent,
    OpenAIResponseInputMessageContentFile,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
    OpenAIResponseMCPApprovalRequest,
    OpenAIResponseMCPApprovalResponse,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageContent,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPCall,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseOutputMessageReasoningItem,
    OpenAIResponseOutputMessageWebSearchToolCall,
    OpenAIResponseReasoning,
    OpenAIResponseText,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
    ResponseGuardrailSpec,
    RetrieveFileContentRequest,
    RetrieveFileRequest,
    RunModerationRequest,
    Safety,
)


async def extract_bytes_from_file(file_id: str, files_api: Files) -> bytes:
    """
    Extract raw bytes from file using the Files API.

    :param file_id: The file identifier (e.g., "file-abc123")
    :param files_api: Files API instance
    :returns: Raw file content as bytes
    :raises: ValueError if file cannot be retrieved
    """
    try:
        response = await files_api.openai_retrieve_file_content(RetrieveFileContentRequest(file_id=file_id))
        return bytes(response.body)
    except Exception as e:
        raise ValueError(f"Failed to retrieve file content for file_id '{file_id}': {str(e)}") from e


def generate_base64_ascii_text_from_bytes(raw_bytes: bytes) -> str:
    """
    Converts raw binary bytes into a safe ASCII text representation for URLs

    :param raw_bytes: the actual bytes that represents file content
    :returns: string of utf-8 characters
    """
    return base64.b64encode(raw_bytes).decode("utf-8")


def construct_data_url(ascii_text: str, mime_type: str | None) -> str:
    """
    Construct data url with decoded data inside

    :param ascii_text: ASCII content
    :param mime_type: MIME type of file
    :returns: data url string (eg. data:image/png,base64,%3Ch1%3EHello%2C%20World%21%3C%2Fh1%3E)
    """
    if not mime_type:
        mime_type = "application/octet-stream"

    return f"data:{mime_type};base64,{ascii_text}"


async def convert_chat_choice_to_response_message(
    choice: OpenAIChoice,
    citation_files: dict[str, str] | None = None,
    *,
    message_id: str | None = None,
) -> OpenAIResponseMessage:
    """Convert an OpenAI Chat Completion choice into an OpenAI Response output message."""
    output_content = choice.message.content or ""

    annotations, clean_text = _extract_citations_from_text(output_content, citation_files or {})
    logprobs = choice.logprobs.content if choice.logprobs and choice.logprobs.content else []

    return OpenAIResponseMessage(
        id=message_id or f"msg_{uuid.uuid4()}",
        content=[
            OpenAIResponseOutputMessageContentOutputText(
                text=clean_text,
                annotations=list(annotations),
                logprobs=logprobs,
            )
        ],
        status="completed",
        role="assistant",
    )


async def _build_tool_result_messages(
    input_item: OpenAIResponseInputFunctionToolCallOutput,
    files_api: Files | None,
) -> list[OpenAIMessageParam]:
    """Convert a function_call_output into chat messages.

    OpenAIToolMessageParam only accepts text content, so image parts are
    placed in a follow-up user message where vision models can see them.
    """
    output = input_item.output
    if not isinstance(output, list):
        return [OpenAIToolMessageParam(content=output, tool_call_id=input_item.call_id)]

    converted = await convert_response_content_to_chat_content(output, files_api=files_api)
    if not isinstance(converted, list):
        return [OpenAIToolMessageParam(content=converted, tool_call_id=input_item.call_id)]

    text_parts: list[OpenAIChatCompletionContentPartTextParam] = []
    image_parts: list[OpenAIChatCompletionContentPartParam] = []
    for part in converted:
        if isinstance(part, OpenAIFile):
            text_parts.append(_file_part_to_text_part(part))
        elif isinstance(part, OpenAIChatCompletionContentPartImageParam):
            image_parts.append(part)
        else:
            text_parts.append(part)

    messages: list[OpenAIMessageParam] = [
        OpenAIToolMessageParam(
            content=text_parts or [OpenAIChatCompletionContentPartTextParam(text="[image]")],
            tool_call_id=input_item.call_id,
        )
    ]
    if image_parts:
        messages.append(OpenAIUserMessageParam(content=image_parts))
    return messages


def _file_part_to_text_part(part: OpenAIFile) -> OpenAIChatCompletionContentPartTextParam:
    # OpenAIFile carries a data URL in part.file.file_data.  Decode text MIME types back
    # to a plain string so tool message content (text-only) stays readable to the model.
    data_url = part.file.file_data if part.file else None
    if data_url and data_url.startswith("data:"):
        # data:<mime>;base64,<payload>
        header, _, payload = data_url.partition(",")
        mime = header.split(";")[0][len("data:") :]
        if mime.startswith("text/") and payload:
            try:
                return OpenAIChatCompletionContentPartTextParam(text=base64.b64decode(payload).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                pass
        if payload:
            return OpenAIChatCompletionContentPartTextParam(text=data_url)
    return OpenAIChatCompletionContentPartTextParam(text=data_url or "")


async def convert_response_content_to_chat_content(
    content: str | Sequence[OpenAIResponseInputMessageContent | OpenAIResponseOutputMessageContent],
    files_api: Files | None,
) -> str | list[OpenAIChatCompletionContentPartParam]:
    """
    Convert the content parts from an OpenAI Response API request into OpenAI Chat Completion content parts.

    The content schemas of each API look similar, but are not exactly the same.

    :param content: The content to convert
    :param files_api: Files API for resolving file_id to raw file content (required if content contains files/images)
    """
    if isinstance(content, str):
        return content

    # Type with union to avoid list invariance issues
    converted_parts: list[OpenAIChatCompletionContentPartParam] = []
    for content_part in content:
        if isinstance(content_part, OpenAIResponseInputMessageContentText):
            converted_parts.append(OpenAIChatCompletionContentPartTextParam(text=content_part.text))
        elif isinstance(content_part, OpenAIResponseOutputMessageContentOutputText):
            converted_parts.append(OpenAIChatCompletionContentPartTextParam(text=content_part.text))
        elif isinstance(content_part, OpenAIResponseInputMessageContentImage):
            detail = content_part.detail
            image_mime_type = None
            if content_part.image_url:
                image_url = OpenAIImageURL(url=content_part.image_url, detail=detail)
                converted_parts.append(OpenAIChatCompletionContentPartImageParam(image_url=image_url))
            elif content_part.file_id:
                if files_api is None:
                    raise ValueError("file_ids are not supported by this implementation of the Stack")
                image_file_response = await files_api.openai_retrieve_file(
                    RetrieveFileRequest(file_id=content_part.file_id)
                )
                if image_file_response.filename:
                    image_mime_type, _ = mimetypes.guess_type(image_file_response.filename)
                raw_image_bytes = await extract_bytes_from_file(content_part.file_id, files_api)
                ascii_text = generate_base64_ascii_text_from_bytes(raw_image_bytes)
                image_data_url = construct_data_url(ascii_text, image_mime_type)
                image_url = OpenAIImageURL(url=image_data_url, detail=detail)
                converted_parts.append(OpenAIChatCompletionContentPartImageParam(image_url=image_url))
            else:
                raise ValueError(
                    f"Image content must have either 'image_url' or 'file_id'. "
                    f"Got image_url={content_part.image_url}, file_id={content_part.file_id}"
                )
        elif isinstance(content_part, OpenAIResponseInputMessageContentFile):
            resolved_file_data = None
            file_data = content_part.file_data
            file_id = content_part.file_id
            file_url = content_part.file_url
            filename = content_part.filename
            file_mime_type = None
            if not any([file_data, file_id, file_url]):
                raise ValueError(
                    f"File content must have at least one of 'file_data', 'file_id', or 'file_url'. "
                    f"Got file_data={file_data}, file_id={file_id}, file_url={file_url}"
                )
            if file_id:
                if files_api is None:
                    raise ValueError("file_ids are not supported by this implementation of the Stack")

                file_response = await files_api.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))
                if not filename:
                    filename = file_response.filename
                file_mime_type, _ = mimetypes.guess_type(file_response.filename)
                raw_file_bytes = await extract_bytes_from_file(file_id, files_api)
                ascii_text = generate_base64_ascii_text_from_bytes(raw_file_bytes)
                resolved_file_data = construct_data_url(ascii_text, file_mime_type)
            elif file_data:
                if file_data.startswith("data:"):
                    resolved_file_data = file_data
                else:
                    # Raw base64 data, wrap in data URL format
                    if filename:
                        file_mime_type, _ = mimetypes.guess_type(filename)
                    resolved_file_data = construct_data_url(file_data, file_mime_type)
            elif file_url:
                resolved_file_data = file_url
            converted_parts.append(
                OpenAIFile(
                    file=OpenAIFileFile(
                        file_data=resolved_file_data,
                        filename=filename,
                    )
                )
            )
        elif isinstance(content_part, str):
            converted_parts.append(OpenAIChatCompletionContentPartTextParam(text=content_part))
        else:
            raise ValueError(
                f"OGX OpenAI Responses does not yet support content type '{type(content_part)}' in this context"
            )
    return converted_parts


async def convert_response_input_to_chat_messages(
    input: str | list[OpenAIResponseInput],
    previous_messages: list[OpenAIMessageParam] | None = None,
    files_api: Files | None = None,
) -> list[OpenAIMessageParam]:
    """
    Convert the input from an OpenAI Response API request into OpenAI Chat Completion messages.

    :param input: The input to convert
    :param previous_messages: Optional previous messages to check for function_call references
    :param files_api: Files API for resolving file_id to raw file content (optional, required for file/image content)
    """
    messages: list[OpenAIMessageParam] = []
    if isinstance(input, list):
        # extract all OpenAIResponseInputFunctionToolCallOutput items
        # so their corresponding OpenAIToolMessageParam instances can
        # be added immediately following the corresponding
        # OpenAIAssistantMessageParam
        tool_call_results: dict[str, list[OpenAIMessageParam]] = {}
        for input_item in input:
            if isinstance(input_item, OpenAIResponseInputFunctionToolCallOutput):
                tool_call_results[input_item.call_id] = await _build_tool_result_messages(input_item, files_api)

        for i, input_item in enumerate(input):
            if isinstance(input_item, OpenAIResponseInputFunctionToolCallOutput):
                # skip as these have been extracted and inserted in order
                pass
            elif isinstance(input_item, OpenAIResponseOutputMessageReasoningItem):
                # skip — reasoning items are consumed by the next assistant message via look-back
                pass
            elif isinstance(input_item, OpenAIResponseOutputMessageFunctionToolCall):
                tool_call = OpenAIChatCompletionToolCall(
                    index=0,
                    id=input_item.call_id,
                    function=OpenAIChatCompletionToolCallFunction(
                        name=input_item.name,
                        arguments=input_item.arguments,
                    ),
                )
                reasoning = _get_preceding_reasoning(input, i)
                if reasoning:
                    msg = AssistantMessageWithReasoning(tool_calls=[tool_call], reasoning_content=reasoning)
                else:
                    msg = OpenAIAssistantMessageParam(tool_calls=[tool_call])  # type: ignore[assignment]
                messages.append(msg)
                if input_item.call_id in tool_call_results:
                    messages.extend(tool_call_results[input_item.call_id])
                    del tool_call_results[input_item.call_id]
            elif isinstance(input_item, OpenAIResponseOutputMessageMCPCall):
                tool_call = OpenAIChatCompletionToolCall(
                    index=0,
                    id=input_item.id,
                    function=OpenAIChatCompletionToolCallFunction(
                        name=input_item.name,
                        arguments=input_item.arguments,
                    ),
                )
                reasoning = _get_preceding_reasoning(input, i)
                if reasoning:
                    msg = AssistantMessageWithReasoning(tool_calls=[tool_call], reasoning_content=reasoning)
                else:
                    msg = OpenAIAssistantMessageParam(tool_calls=[tool_call])  # type: ignore[assignment]
                messages.append(msg)
                # Output can be None, use empty string as fallback
                output_content = input_item.output if input_item.output is not None else ""
                messages.append(
                    OpenAIToolMessageParam(
                        content=output_content,
                        tool_call_id=input_item.id,
                    )
                )
            elif isinstance(input_item, OpenAIResponseOutputMessageMCPListTools):
                # the tool list will be handled separately
                pass
            elif isinstance(
                input_item,
                OpenAIResponseOutputMessageWebSearchToolCall | OpenAIResponseOutputMessageFileSearchToolCall,
            ):
                # these tool calls are tracked internally but not converted to chat messages
                pass
            elif isinstance(input_item, OpenAIResponseMCPApprovalRequest) or isinstance(
                input_item, OpenAIResponseMCPApprovalResponse
            ):
                # these are handled by the responses impl itself and not pass through to chat completions
                pass
            elif isinstance(input_item, OpenAIResponseCompaction):
                # Convert compaction summary to an assistant message so the model sees prior context
                messages.append(OpenAIAssistantMessageParam(content=input_item.encrypted_content))
            elif isinstance(input_item, OpenAIResponseMessage):
                # Narrow type to OpenAIResponseMessage which has content and role attributes
                content = await convert_response_content_to_chat_content(input_item.content, files_api)
                message_type = await get_message_type_by_role(input_item.role)
                if message_type is None:
                    raise ValueError(
                        f"OGX OpenAI Responses does not yet support message role '{input_item.role}' in this context"
                    )
                # Skip user messages that duplicate the last user message in previous_messages
                # This handles cases where input includes context for function_call_outputs
                if previous_messages and input_item.role == "user":
                    last_user_msg = None
                    for prev_msg in reversed(previous_messages):
                        if isinstance(prev_msg, OpenAIUserMessageParam):
                            last_user_msg = prev_msg
                            break
                    if last_user_msg:
                        last_user_content = getattr(last_user_msg, "content", None)
                        if last_user_content == content:
                            continue  # Skip duplicate user message
                # Attach preceding reasoning to assistant messages
                if input_item.role == "assistant":
                    reasoning = _get_preceding_reasoning(input, i)
                    if reasoning:
                        messages.append(AssistantMessageWithReasoning(content=content, reasoning_content=reasoning))  # type: ignore[arg-type]
                        continue
                # Dynamic message type call - different message types have different content expectations
                messages.append(message_type(content=content))  # type: ignore[call-arg,arg-type]
            else:
                # Fail loudly on unknown types so future additions to
                # OpenAIResponseInput don't silently drop data.
                raise ValueError(f"Unexpected input item type: {type(input_item).__name__}")
        if len(tool_call_results):
            # Check if unpaired function_call_outputs reference function_calls from previous messages
            if previous_messages:
                previous_call_ids = _extract_tool_call_ids(previous_messages)
                for call_id in list(tool_call_results.keys()):
                    if call_id in previous_call_ids:
                        # Valid: this output references a call from previous messages
                        # Add the tool message(s) — may include a follow-up user message for images
                        messages.extend(tool_call_results[call_id])
                        del tool_call_results[call_id]

            # If still have unpaired outputs, error
            if len(tool_call_results):
                raise ValueError(
                    f"Received function_call_output(s) with call_id(s) {tool_call_results.keys()}, but no corresponding function_call"
                )
    else:
        messages.append(OpenAIUserMessageParam(content=input))
    return messages


def _extract_tool_call_ids(messages: list[OpenAIMessageParam]) -> set[str]:
    """Extract all tool_call IDs from messages."""
    call_ids = set()
    for msg in messages:
        if isinstance(msg, OpenAIAssistantMessageParam):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    # tool_call is a Pydantic model, use attribute access
                    call_ids.add(tool_call.id)
    return call_ids


def _get_preceding_reasoning(input_items: list[OpenAIResponseInput], index: int) -> str | None:
    """If the item immediately before current index is a reasoning item, return its text."""
    if index > 0:
        preceding = input_items[index - 1]
        if isinstance(preceding, OpenAIResponseOutputMessageReasoningItem) and preceding.content:
            return " ".join(c.text for c in preceding.content)
    return None


async def convert_response_text_to_chat_response_format(
    text: OpenAIResponseText,
) -> OpenAIResponseFormatParam:
    """
    Convert an OpenAI Response text parameter into an OpenAI Chat Completion response format.
    """
    if not text.format or text.format["type"] == "text":
        return OpenAIResponseFormatText(type="text")
    if text.format["type"] == "json_object":
        return OpenAIResponseFormatJSONObject()
    if text.format["type"] == "json_schema":
        # Assert name exists for json_schema format
        assert text.format.get("name"), "json_schema format requires a name"
        schema_name: str = text.format["name"]  # type: ignore[assignment]
        return OpenAIResponseFormatJSONSchema(
            json_schema=OpenAIJSONSchema(name=schema_name, schema=text.format["schema"])
        )
    raise ValueError(f"Unsupported text format: {text.format}")


async def get_message_type_by_role(role: str) -> type[OpenAIMessageParam] | None:
    """Get the appropriate OpenAI message parameter type for a given role."""
    role_to_type = {
        "user": OpenAIUserMessageParam,
        "system": OpenAISystemMessageParam,
        "assistant": OpenAIAssistantMessageParam,
        "developer": OpenAIDeveloperMessageParam,
    }
    return role_to_type.get(role)  # type: ignore[return-value]  # Pydantic models use ModelMetaclass


def _extract_citations_from_text(
    text: str, citation_files: dict[str, str]
) -> tuple[list[OpenAIResponseAnnotationFileCitation], str]:
    """Extract citation markers from text and create annotations

    Args:
        text: The text containing citation markers like [file-Cn3MSNn72ENTiiq11Qda4A]
        citation_files: Dictionary mapping file_id to filename

    Returns:
        Tuple of (annotations_list, clean_text_without_markers)
    """
    file_id_regex = re.compile(r"<\|(?P<file_id>file-[A-Za-z0-9_-]+)\|>")

    annotations = []
    parts = []
    total_len = 0
    last_end = 0

    for m in file_id_regex.finditer(text):
        # segment before the marker
        prefix = text[last_end : m.start()]

        # drop one space if it exists (since marker is at sentence end)
        if prefix.endswith(" "):
            prefix = prefix[:-1]

        parts.append(prefix)
        total_len += len(prefix)

        fid = m.group(1)
        if fid in citation_files:
            annotations.append(
                OpenAIResponseAnnotationFileCitation(
                    file_id=fid,
                    filename=citation_files[fid],
                    index=total_len,  # index points to punctuation
                )
            )

        last_end = m.end()

    parts.append(text[last_end:])
    cleaned_text = "".join(parts)
    return annotations, cleaned_text


def is_function_tool_call(
    tool_call: OpenAIChatCompletionToolCall,
    tools: list[OpenAIResponseInputTool],
) -> bool:
    """Check whether a tool call corresponds to a user-defined function tool.

    Args:
        tool_call: the tool call to check
        tools: list of available response input tools

    Returns:
        True if the tool call matches a function tool in the tools list
    """
    if not tool_call.function:
        return False
    for t in tools:
        if t.type == "function" and t.name == tool_call.function.name:
            return True
    return False


async def run_guardrails(safety_api: Safety | None, messages: str, guardrail_ids: list[str]) -> str | None:
    """Run guardrails against messages and return violation message if blocked."""
    if not messages:
        return None

    # If safety API is not available, skip guardrails
    if safety_api is None:
        return None

    # Look up shields to get their provider_resource_id (actual model ID)
    model_ids = []
    # TODO: list_shields not in Safety interface but available at runtime via API routing
    shields_list = await safety_api.routing_table.list_shields()  # type: ignore[attr-defined]

    for guardrail_id in guardrail_ids:
        matching_shields = [shield for shield in shields_list.data if shield.identifier == guardrail_id]
        if matching_shields:
            model_id = matching_shields[0].provider_resource_id
            model_ids.append(model_id)
        else:
            raise ValueError(f"No shield found with identifier '{guardrail_id}'")

    guardrail_tasks = [
        safety_api.run_moderation(RunModerationRequest(input=messages, model=model_id)) for model_id in model_ids
    ]
    responses = await asyncio.gather(*guardrail_tasks)

    for response in responses:
        for result in response.results:
            if result.flagged:
                message = result.user_message or "Content blocked by safety guardrails"
                flagged_categories = (
                    [cat for cat, flagged in result.categories.items() if flagged] if result.categories else []
                )
                violation_type = result.metadata.get("violation_type", []) if result.metadata else []

                if flagged_categories:
                    message += f" (flagged for: {', '.join(flagged_categories)})"
                if violation_type:
                    message += f" (violation type: {', '.join(violation_type)})"

                return message

    # No violations found
    return None


def extract_guardrail_ids(guardrails: list | None) -> list[str]:
    """Extract guardrail IDs from guardrails parameter, handling both string IDs and ResponseGuardrailSpec objects."""
    if not guardrails:
        return []

    guardrail_ids = []
    for guardrail in guardrails:
        if isinstance(guardrail, str):
            guardrail_ids.append(guardrail)
        elif isinstance(guardrail, ResponseGuardrailSpec):
            guardrail_ids.append(guardrail.type)
        else:
            raise ValueError(f"Unknown guardrail format: {guardrail}, expected str or ResponseGuardrailSpec")

    return guardrail_ids


def convert_mcp_tool_choice(
    chat_tool_names: list[str],
    server_label: str | None = None,
    server_label_to_tools: dict[str, list[str]] | None = None,
    tool_name: str | None = None,
) -> dict[str, str | dict[str, str]] | list[dict[str, str | dict[str, str]]] | None:
    """Convert a responses tool choice of type mcp to a chat completions compatible function tool choice."""

    if tool_name:
        if tool_name not in chat_tool_names:
            return None
        return {"type": "function", "function": {"name": tool_name}}

    elif server_label and server_label_to_tools:
        # no tool name specified, so we need to enforce an allowed_tools with the function tools derived only from the given server label
        # Use reverse mapping for lookup by server_label
        # This already accounts for allowed_tools restrictions applied during _process_mcp_tool
        tool_names = server_label_to_tools.get(server_label, [])
        if not tool_names:
            return None
        matching_tools: list[dict[str, str | dict[str, str]]] = [
            {"type": "function", "function": {"name": name}} for name in tool_names
        ]
        return matching_tools
    return []


logger = get_logger("llama_stack.providers.inline.responses.builtin.responses.utils", category="agents::builtin")


def should_summarize_reasoning(reasoning: OpenAIResponseReasoning | None) -> bool:
    """Check whether reasoning summaries were requested."""
    return bool(reasoning and reasoning.summary)


def build_summary_prompt(reasoning_text: str, summary_mode: str) -> str:
    """Build the prompt for the reasoning summarization inference call."""
    if summary_mode == "detailed":
        return (
            "Summarize the following chain-of-thought reasoning. "
            "Preserve the key logical steps, decisions, and conclusions. "
            "Be thorough but remove redundancy.\n\n"
            f"Reasoning:\n{reasoning_text}"
        )
    return (
        "Summarize the following chain-of-thought reasoning in one or two sentences. "
        "Focus only on the final conclusion and the most important reasoning step.\n\n"
        f"Reasoning:\n{reasoning_text}"
    )


async def summarize_reasoning(
    inference_api: Inference,
    model: str,
    reasoning_text: str,
    summary_mode: str,
    summary_usage: list[OpenAIChatCompletionUsage] | None = None,
) -> str | None:
    """Make a second inference call to summarize reasoning content."""
    prompt_text = build_summary_prompt(reasoning_text, summary_mode)

    summary_params = OpenAIChatCompletionRequestWithExtraBody(
        model=model,
        messages=[
            OpenAISystemMessageParam(
                content="You are a helpful assistant that summarizes reasoning traces.",
            ),
            OpenAIUserMessageParam(content=prompt_text),
        ],
        stream=False,
        temperature=0.3,
    )

    try:
        summary_result = await inference_api.openai_chat_completion(summary_params)
    except Exception:
        logger.exception("Failed to generate reasoning summary")
        raise

    if isinstance(summary_result, AsyncIterator):
        raise RuntimeError("Expected non-streaming response from summary call")

    if summary_usage is not None and summary_result.usage:
        summary_usage.append(summary_result.usage)

    summary_text = summary_result.choices[0].message.content if summary_result.choices else None
    if not summary_text:
        return None

    return summary_text
