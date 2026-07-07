# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import time
from html import escape
from typing import Any

from fastapi import UploadFile

from ogx.core.request_headers import get_authenticated_user
from ogx.log import get_logger
from ogx.providers.inline.responses.builtin.config import MemoryConfig
from ogx.providers.utils.responses.responses_store import ResponsesStore
from ogx_api import (
    DeleteFileRequest,
    Files,
    Inference,
    OpenAIFileUploadPurpose,
    OpenAIMessageParam,
    OpenAIResponseInput,
    OpenAIResponseMessage,
    OpenAIUserMessageParam,
    UploadFileRequest,
    VectorIO,
    VectorStoreNotFoundError,
)
from ogx_api.inference import OpenAIChatCompletionRequestWithExtraBody
from ogx_api.responses.models import MemoryToolConfig
from ogx_api.vector_io.models import OpenAIAttachFileRequest, OpenAISearchVectorStoreRequest, VectorStoreSearchResponse

from .utils import APPROX_CHARS_PER_TOKEN

logger = get_logger(name=__name__, category="openai_responses::memory")

_TRUNCATION_SUFFIX = "[truncated]"


def extract_memory_query(input: str | list[OpenAIResponseInput]) -> str | None:
    """Extract the latest user text from the current request input."""
    if isinstance(input, str):
        return input if input.strip() else None

    for item in reversed(input):
        if not isinstance(item, OpenAIResponseMessage) or item.role != "user":
            continue

        text_segments = _extract_text_from_message(item)
        if text_segments:
            return "\n".join(text_segments)

    return None


def build_memory_filters(
    memory_config: MemoryConfig,
    owner_id: str | None,
    request_filters: dict[str, Any] | None,
) -> dict[str, Any]:
    if not owner_id:
        raise ValueError("Failed to build memory filters: owner is required")

    filters: list[dict[str, Any]] = [
        {"type": "eq", "key": memory_config.memory_metadata_key, "value": True},
        {"type": "eq", "key": memory_config.owner_metadata_key, "value": owner_id},
    ]
    if request_filters:
        filters.append(request_filters)

    return {"type": "and", "filters": filters}


async def resolve_memory_context(
    vector_io_api: VectorIO,
    memory_config: MemoryConfig,
    request_memory: MemoryToolConfig | None,
    input: str | list[OpenAIResponseInput],
    metadata: dict[str, str] | None,
    safety_identifier: str | None,
) -> str | None:
    if not memory_config.enabled:
        return None
    if request_memory is None:
        logger.debug("Skipping memory retrieval", reason="missing request opt-in")
        return None
    if not request_memory.enabled:
        return None

    vector_store_id = (
        request_memory.vector_store_id
        if request_memory.vector_store_id is not None
        else memory_config.default_vector_store_id
    )
    if not vector_store_id:
        logger.debug("Skipping memory retrieval", reason="missing vector store")
        return None

    owner_id = resolve_memory_owner_id(request_memory, metadata, safety_identifier)
    if not owner_id:
        logger.debug("Skipping memory retrieval", reason="missing owner")
        return None

    filters = build_memory_filters(memory_config, owner_id, request_memory.filters)
    max_num_results = (
        request_memory.max_num_results if request_memory.max_num_results is not None else memory_config.max_num_results
    )
    max_context_tokens = (
        request_memory.max_context_tokens
        if request_memory.max_context_tokens is not None
        else memory_config.max_context_tokens
    )
    ranking_options = request_memory.ranking_options
    query = extract_memory_query(input)
    if query is None:
        logger.debug("Skipping memory retrieval", reason="missing query text")
        return None

    try:
        search_response = await vector_io_api.openai_search_vector_store(
            vector_store_id=vector_store_id,
            request=OpenAISearchVectorStoreRequest(
                query=query,
                filters=filters,
                max_num_results=max_num_results,
                ranking_options=ranking_options,
                rewrite_query=False,
                search_mode="hybrid",
            ),
        )
    except VectorStoreNotFoundError as exc:
        logger.warning(
            "Failed to retrieve memory context because vector store was not found",
            vector_store_id=vector_store_id,
            error=str(exc),
        )
        return None

    if not search_response.data:
        return None

    return _format_memory_context(
        memory_config=memory_config,
        results=search_response.data,
        max_context_tokens=max_context_tokens,
    )


async def write_conversation_memory(
    inference_api: Inference,
    files_api: Files,
    vector_io_api: VectorIO,
    responses_store: ResponsesStore,
    memory_config: MemoryConfig,
    request_memory: MemoryToolConfig | None,
    conversation_id: str | None,
    response_id: str,
    response_status: str | None,
    model: str | None,
    metadata: dict[str, str] | None,
    safety_identifier: str | None,
    owner_id: str | None = None,
) -> None:
    if (
        not memory_config.enabled
        or not memory_config.write_enabled
        or response_status != "completed"
        or not conversation_id
    ):
        return
    if request_memory is not None and not request_memory.enabled:
        return

    vector_store_id = (
        request_memory.vector_store_id
        if request_memory is not None and request_memory.vector_store_id is not None
        else memory_config.default_vector_store_id
    )
    if not vector_store_id:
        logger.debug("Skipping memory write", reason="missing vector store")
        return

    owner_id = _normalize_owner_id(owner_id) or resolve_memory_owner_id(request_memory, metadata, safety_identifier)
    if not owner_id:
        logger.debug("Skipping memory write", reason="missing owner")
        return

    summarization_model = memory_config.summarization_model or model
    if not summarization_model:
        logger.warning(
            "Failed to write memory summary",
            reason="missing summarization model",
            conversation_id=conversation_id,
            response_id=response_id,
        )
        return

    messages = await responses_store.get_conversation_messages(conversation_id)
    if not messages:
        logger.debug("Skipping memory write without conversation messages", conversation_id=conversation_id)
        return

    summary_text = await _generate_memory_summary(
        inference_api=inference_api,
        model=summarization_model,
        messages=messages,
        summarization_prompt=memory_config.summarization_prompt,
    )
    if not summary_text:
        logger.debug("Skipping memory write without summary text", conversation_id=conversation_id)
        return

    created_at = int(time.time())
    markdown = _format_memory_markdown(
        summary_text=summary_text,
        messages=messages,
        owner_id=owner_id,
        conversation_id=conversation_id,
        response_id=response_id,
        created_at=created_at,
    )
    markdown_bytes = markdown.encode("utf-8")
    upload = UploadFile(
        file=io.BytesIO(markdown_bytes),
        filename=f"{conversation_id}.memory.md",
        size=len(markdown_bytes),
    )
    uploaded_file = await files_api.openai_upload_file(
        request=UploadFileRequest(purpose=OpenAIFileUploadPurpose.ASSISTANTS),
        file=upload,
    )

    attached_file = await vector_io_api.openai_attach_file_to_vector_store(
        vector_store_id=vector_store_id,
        request=OpenAIAttachFileRequest(
            file_id=uploaded_file.id,
            attributes={
                memory_config.memory_metadata_key: True,
                memory_config.owner_metadata_key: owner_id,
                "conversation_id": conversation_id,
                "response_id": response_id,
                "created_at": float(created_at),
            },
        ),
    )
    if getattr(attached_file, "status", None) == "failed":
        logger.warning(
            "Failed to write memory summary",
            reason="vector store attachment failed",
            vector_store_id=vector_store_id,
            file_id=uploaded_file.id,
        )
        await _delete_uploaded_memory_file(files_api=files_api, file_id=uploaded_file.id)
        return

    previous_record = await responses_store.get_memory_record(
        owner_id=owner_id,
        conversation_id=conversation_id,
        vector_store_id=vector_store_id,
    )
    await responses_store.upsert_memory_record(
        owner_id=owner_id,
        conversation_id=conversation_id,
        vector_store_id=vector_store_id,
        file_id=uploaded_file.id,
        response_id=response_id,
    )

    if previous_record is not None and previous_record.file_id != uploaded_file.id:
        await _delete_previous_memory_file(
            files_api=files_api,
            vector_io_api=vector_io_api,
            vector_store_id=vector_store_id,
            file_id=previous_record.file_id,
        )


def resolve_memory_owner_id(
    request_memory: MemoryToolConfig | None,
    metadata: dict[str, str] | None,
    safety_identifier: str | None,
) -> str | None:
    user = get_authenticated_user()
    if user is not None:
        return _normalize_owner_id(user.principal)
    if request_memory is not None:
        owner_id = _normalize_owner_id(request_memory.owner_id)
        if owner_id:
            return owner_id
    owner_id = _normalize_owner_id(safety_identifier)
    if owner_id:
        return owner_id
    if metadata:
        return _normalize_owner_id(metadata.get("owner_id")) or _normalize_owner_id(metadata.get("user_id"))
    return None


async def _delete_previous_memory_file(
    files_api: Files,
    vector_io_api: VectorIO,
    vector_store_id: str,
    file_id: str,
) -> None:
    try:
        await vector_io_api.openai_delete_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )
    except Exception as exc:
        logger.warning(
            "Failed to delete previous memory file from vector store",
            vector_store_id=vector_store_id,
            file_id=file_id,
            error=str(exc),
        )

    await _delete_uploaded_memory_file(files_api=files_api, file_id=file_id)


async def _delete_uploaded_memory_file(files_api: Files, file_id: str) -> None:
    try:
        await files_api.openai_delete_file(request=DeleteFileRequest(file_id=file_id))
    except Exception as exc:
        logger.warning(
            "Failed to delete memory file",
            file_id=file_id,
            error=str(exc),
        )


def _normalize_owner_id(owner_id: str | None) -> str | None:
    if owner_id is None:
        return None
    owner_id = owner_id.strip()
    return owner_id or None


async def _generate_memory_summary(
    inference_api: Inference,
    model: str,
    messages: list[OpenAIMessageParam],
    summarization_prompt: str,
) -> str:
    summary_messages = list(messages)
    summary_messages.append(OpenAIUserMessageParam(role="user", content=summarization_prompt))
    completion = await inference_api.openai_chat_completion(
        OpenAIChatCompletionRequestWithExtraBody(
            model=model,
            messages=summary_messages,
            stream=False,
        )
    )

    if hasattr(completion, "choices") and completion.choices:
        choice = completion.choices[0]
        if choice.message and choice.message.content:
            return choice.message.content
    return ""


def _format_memory_markdown(
    summary_text: str,
    messages: list[OpenAIMessageParam],
    owner_id: str,
    conversation_id: str,
    response_id: str,
    created_at: int,
) -> str:
    transcript = _format_memory_transcript(messages)
    return (
        "# Conversation Memory\n\n"
        f"- owner_id: {owner_id}\n"
        f"- conversation_id: {conversation_id}\n"
        f"- response_id: {response_id}\n"
        f"- created_at: {created_at}\n\n"
        "## Summary\n\n"
        f"{summary_text.strip()}\n"
        "\n## Searchable Conversation Transcript\n\n"
        f"{transcript}\n"
    )


def _format_memory_transcript(messages: list[OpenAIMessageParam]) -> str:
    lines: list[str] = []
    for message in messages:
        text = _message_content_to_text(message.content).strip()
        if text:
            lines.append(f"- {message.role}: {text}")
    return "\n".join(lines)


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    text_segments: list[str] = []
    for content_item in content:
        text = content_item.get("text") if isinstance(content_item, dict) else getattr(content_item, "text", None)
        if isinstance(text, str) and text:
            text_segments.append(text)
    return "\n".join(text_segments)


def _extract_text_from_message(message: OpenAIResponseMessage) -> list[str]:
    if isinstance(message.content, str):
        return [message.content] if message.content.strip() else []

    text_segments: list[str] = []
    for content_item in message.content:
        text = getattr(content_item, "text", None)
        if isinstance(text, str) and text.strip():
            text_segments.append(text)
    return text_segments


def _format_memory_context(
    memory_config: MemoryConfig,
    results: list[VectorStoreSearchResponse],
    max_context_tokens: int,
) -> str | None:
    header = memory_config.read_prompt_template.strip()
    opening = f"{header}\n\n<memories>"
    closing = "</memories>"
    snippets: list[str] = []
    used_chars = len(opening) + len(closing) + 1

    for result in results:
        snippet = _format_memory_result(len(snippets) + 1, result)
        candidate_chars = used_chars + len(snippet) + 1
        if _estimate_tokens_for_chars(candidate_chars) <= max_context_tokens:
            snippets.append(snippet)
            used_chars = candidate_chars
            continue

        if not snippets:
            remaining_chars = max_context_tokens * APPROX_CHARS_PER_TOKEN - used_chars - 1
            snippets.append(_truncate_text(snippet, remaining_chars))
        break

    if not snippets:
        return None

    return "\n".join([opening, *snippets, closing])


def _format_memory_result(index: int, result: VectorStoreSearchResponse) -> str:
    attributes = result.attributes or {}
    created_at = attributes.get("created_at", "")
    text = "\n".join(
        text for content in result.content if isinstance(text := getattr(content, "text", None), str) and text
    )
    escaped_file_id = _escape_xml_attr(result.file_id)
    escaped_created_at = _escape_xml_attr(created_at)
    escaped_text = _escape_xml_text(text)
    return (
        f'<memory index="{index}" file_id="{escaped_file_id}" created_at="{escaped_created_at}">\n'
        f"{escaped_text}\n"
        "</memory>"
    )


def _escape_xml_attr(value: Any) -> str:
    return escape(str(value), quote=True)


def _escape_xml_text(value: str) -> str:
    return escape(value, quote=False)


def _estimate_tokens_for_chars(char_count: int) -> int:
    return max(1, char_count // APPROX_CHARS_PER_TOKEN)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= len(_TRUNCATION_SUFFIX):
        return _TRUNCATION_SUFFIX.strip()
    return text[: max_chars - len(_TRUNCATION_SUFFIX)].rstrip() + _TRUNCATION_SUFFIX
