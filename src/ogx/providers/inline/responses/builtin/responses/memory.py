# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import io
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from html import escape
from typing import Any

from fastapi import UploadFile

from ogx.core.datatypes import User
from ogx.core.request_headers import PROVIDER_DATA_VAR, RequestProviderDataContext, get_authenticated_user
from ogx.log import get_logger
from ogx.providers.inline.responses.builtin.config import MemoryConfig
from ogx.providers.utils.responses.responses_store import ResponsesStore
from ogx_api import (
    DeleteFileRequest,
    Files,
    Inference,
    OpenAICreateVectorStoreRequestWithExtraBody,
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
_DEFAULT_MEMORY_VECTOR_STORE_LOCKS: dict[str, asyncio.Lock] = {}
_DEFAULT_MEMORY_VECTOR_STORE_WAIT_ATTEMPTS = 100
_DEFAULT_MEMORY_VECTOR_STORE_WAIT_SECONDS = 0.05
_DEFAULT_MEMORY_VECTOR_STORE_STALE_CLAIM_SECONDS = 300


@dataclass(frozen=True)
class MemoryVectorStoreReference:
    """Resolved memory vector store and whether it is an internal backing store."""

    vector_store_id: str
    internal: bool


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
    responses_store: ResponsesStore | None = None,
) -> str | None:
    if not memory_config.enabled:
        return None
    if request_memory is not None and not request_memory.enabled:
        return None

    owner_id = resolve_memory_owner_id(request_memory, metadata, safety_identifier)
    if not owner_id:
        logger.debug("Skipping memory retrieval", reason="missing owner")
        return None

    vector_store = await resolve_memory_vector_store(
        vector_io_api=vector_io_api,
        responses_store=responses_store,
        memory_config=memory_config,
        request_memory=request_memory,
        owner_id=owner_id,
    )
    if vector_store is None:
        logger.debug("Skipping memory retrieval", reason="missing vector store")
        return None

    request_filters = request_memory.filters if request_memory is not None else None
    filters = build_memory_filters(memory_config, owner_id, request_filters)
    max_num_results = (
        request_memory.max_num_results
        if request_memory is not None and request_memory.max_num_results is not None
        else memory_config.max_num_results
    )
    max_context_tokens = (
        request_memory.max_context_tokens
        if request_memory is not None and request_memory.max_context_tokens is not None
        else memory_config.max_context_tokens
    )
    ranking_options = request_memory.ranking_options if request_memory is not None else None
    query = extract_memory_query(input)
    if query is None:
        logger.debug("Skipping memory retrieval", reason="missing query text")
        return None

    try:
        with _memory_access_context(memory_config, vector_store.internal):
            search_response = await vector_io_api.openai_search_vector_store(
                vector_store_id=vector_store.vector_store_id,
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
            vector_store_id=vector_store.vector_store_id,
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


async def resolve_memory_vector_store_id(
    vector_io_api: VectorIO,
    responses_store: ResponsesStore | None,
    memory_config: MemoryConfig,
    request_memory: MemoryToolConfig | None,
    owner_id: str,
) -> str | None:
    vector_store = await resolve_memory_vector_store(
        vector_io_api=vector_io_api,
        responses_store=responses_store,
        memory_config=memory_config,
        request_memory=request_memory,
        owner_id=owner_id,
    )
    return vector_store.vector_store_id if vector_store is not None else None


async def resolve_memory_vector_store(
    vector_io_api: VectorIO,
    responses_store: ResponsesStore | None,
    memory_config: MemoryConfig,
    request_memory: MemoryToolConfig | None,
    owner_id: str,
) -> MemoryVectorStoreReference | None:
    if request_memory is not None and request_memory.vector_store_id:
        return MemoryVectorStoreReference(vector_store_id=request_memory.vector_store_id, internal=False)
    if memory_config.default_vector_store_id:
        return MemoryVectorStoreReference(vector_store_id=memory_config.default_vector_store_id, internal=True)
    if not memory_config.auto_create_default_vector_store:
        return None
    if responses_store is None:
        logger.debug("Skipping default memory vector store creation", reason="missing responses store")
        return None

    namespace = memory_config.default_vector_store_namespace
    with _memory_admin_context(memory_config):
        existing = await responses_store.get_default_memory_vector_store(
            namespace=namespace,
        )
    if existing is not None and existing.vector_store_id is not None:
        return MemoryVectorStoreReference(vector_store_id=existing.vector_store_id, internal=True)

    async with _default_memory_vector_store_lock(namespace):
        with _memory_admin_context(memory_config):
            existing = await responses_store.get_default_memory_vector_store(
                namespace=namespace,
            )
        if existing is not None and existing.vector_store_id is not None:
            return MemoryVectorStoreReference(vector_store_id=existing.vector_store_id, internal=True)
        if _is_stale_default_memory_vector_store_claim(existing):
            with _memory_admin_context(memory_config):
                await responses_store.delete_default_memory_vector_store_claim(namespace=namespace)

        with _memory_admin_context(memory_config):
            claimed = await responses_store.claim_default_memory_vector_store(namespace=namespace)
        if not claimed:
            return await _wait_for_default_memory_vector_store(
                vector_io_api=vector_io_api,
                responses_store=responses_store,
                memory_config=memory_config,
                namespace=namespace,
            )

        return await _create_default_memory_vector_store(
            vector_io_api=vector_io_api,
            responses_store=responses_store,
            memory_config=memory_config,
            namespace=namespace,
        )


async def _create_default_memory_vector_store(
    vector_io_api: VectorIO,
    responses_store: ResponsesStore,
    memory_config: MemoryConfig,
    namespace: str,
) -> MemoryVectorStoreReference:
    try:
        with _memory_admin_context(memory_config):
            vector_store = await vector_io_api.openai_create_vector_store(
                OpenAICreateVectorStoreRequestWithExtraBody.model_validate(
                    {
                        "name": f"ogx-memory-{namespace}",
                        "metadata": {
                            "ogx_memory_store": True,
                            "ogx_memory_namespace": namespace,
                        },
                        "provider_id": memory_config.default_vector_store_provider_id,
                    }
                )
            )

        provider_id = memory_config.default_vector_store_provider_id
        vector_store_metadata = getattr(vector_store, "metadata", None)
        if isinstance(vector_store_metadata, dict):
            provider_id_value = vector_store_metadata.get("provider_id")
            if isinstance(provider_id_value, str):
                provider_id = provider_id_value

        with _memory_admin_context(memory_config):
            await responses_store.upsert_default_memory_vector_store(
                namespace=namespace,
                vector_store_id=vector_store.id,
                provider_id=provider_id,
            )
    except Exception:
        with _memory_admin_context(memory_config):
            await responses_store.delete_default_memory_vector_store_claim(namespace=namespace)
        raise
    return MemoryVectorStoreReference(vector_store_id=vector_store.id, internal=True)


def _is_stale_default_memory_vector_store_claim(record: Any | None) -> bool:
    if record is None or getattr(record, "vector_store_id", None) is not None:
        return False

    updated_at = getattr(record, "updated_at", None)
    if not isinstance(updated_at, int | float):
        return False
    return time.time() - updated_at >= _DEFAULT_MEMORY_VECTOR_STORE_STALE_CLAIM_SECONDS


async def _wait_for_default_memory_vector_store(
    vector_io_api: VectorIO,
    responses_store: ResponsesStore,
    memory_config: MemoryConfig,
    namespace: str,
) -> MemoryVectorStoreReference | None:
    for _ in range(_DEFAULT_MEMORY_VECTOR_STORE_WAIT_ATTEMPTS):
        await asyncio.sleep(_DEFAULT_MEMORY_VECTOR_STORE_WAIT_SECONDS)
        with _memory_admin_context(memory_config):
            existing = await responses_store.get_default_memory_vector_store(namespace=namespace)
        if existing is not None and existing.vector_store_id is not None:
            return MemoryVectorStoreReference(vector_store_id=existing.vector_store_id, internal=True)
        if _is_stale_default_memory_vector_store_claim(existing):
            with _memory_admin_context(memory_config):
                await responses_store.delete_default_memory_vector_store_claim(namespace=namespace)
                claimed = await responses_store.claim_default_memory_vector_store(namespace=namespace)
            if claimed:
                return await _create_default_memory_vector_store(
                    vector_io_api=vector_io_api,
                    responses_store=responses_store,
                    memory_config=memory_config,
                    namespace=namespace,
                )
            logger.warning(
                "Failed to recover stale default memory vector store claim",
                namespace=namespace,
            )
            return None

    logger.warning(
        "Failed to resolve default memory vector store before timeout",
        namespace=namespace,
    )
    return None


def _default_memory_vector_store_lock(namespace: str) -> asyncio.Lock:
    lock = _DEFAULT_MEMORY_VECTOR_STORE_LOCKS.get(namespace)
    if lock is None:
        lock = asyncio.Lock()
        _DEFAULT_MEMORY_VECTOR_STORE_LOCKS[namespace] = lock
    return lock


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

    owner_id = _normalize_owner_id(owner_id) or resolve_memory_owner_id(request_memory, metadata, safety_identifier)
    if not owner_id:
        logger.debug("Skipping memory write", reason="missing owner")
        return

    vector_store = await resolve_memory_vector_store(
        vector_io_api=vector_io_api,
        responses_store=responses_store,
        memory_config=memory_config,
        request_memory=request_memory,
        owner_id=owner_id,
    )
    if vector_store is None:
        logger.debug("Skipping memory write", reason="missing vector store")
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
    bounded_messages = _limit_memory_summary_messages(messages, memory_config.max_summary_messages)

    summary_text = await _generate_memory_summary(
        inference_api=inference_api,
        model=summarization_model,
        messages=bounded_messages,
        summarization_prompt=memory_config.summarization_prompt,
    )
    if not summary_text:
        logger.debug("Skipping memory write without summary text", conversation_id=conversation_id)
        return

    created_at = int(time.time())
    markdown = _format_memory_markdown(
        summary_text=summary_text,
        messages=bounded_messages,
        owner_id=owner_id,
        conversation_id=conversation_id,
        response_id=response_id,
        created_at=created_at,
        max_transcript_chars=memory_config.max_transcript_chars,
    )
    markdown_bytes = markdown.encode("utf-8")
    upload = UploadFile(
        file=io.BytesIO(markdown_bytes),
        filename=f"{conversation_id}.memory.md",
        size=len(markdown_bytes),
    )
    with _memory_access_context(memory_config, vector_store.internal):
        uploaded_file = await files_api.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFileUploadPurpose.ASSISTANTS),
            file=upload,
        )

        attached_file = await vector_io_api.openai_attach_file_to_vector_store(
            vector_store_id=vector_store.vector_store_id,
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
            vector_store_id=vector_store.vector_store_id,
            file_id=uploaded_file.id,
        )
        with _memory_access_context(memory_config, vector_store.internal):
            await _delete_uploaded_memory_file(files_api=files_api, file_id=uploaded_file.id)
        return

    previous_record = await responses_store.get_memory_record(
        owner_id=owner_id,
        conversation_id=conversation_id,
        vector_store_id=vector_store.vector_store_id,
    )
    await responses_store.upsert_memory_record(
        owner_id=owner_id,
        conversation_id=conversation_id,
        vector_store_id=vector_store.vector_store_id,
        file_id=uploaded_file.id,
        response_id=response_id,
    )

    if previous_record is not None and previous_record.file_id != uploaded_file.id:
        await _delete_previous_memory_file(
            files_api=files_api,
            vector_io_api=vector_io_api,
            vector_store_id=vector_store.vector_store_id,
            file_id=previous_record.file_id,
            memory_config=memory_config,
            use_admin_context=vector_store.internal,
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
    memory_config: MemoryConfig,
    use_admin_context: bool,
) -> None:
    with _memory_access_context(memory_config, use_admin_context):
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


def _memory_admin_context(memory_config: MemoryConfig) -> RequestProviderDataContext:
    provider_data = PROVIDER_DATA_VAR.get()
    copied_provider_data = dict(provider_data) if isinstance(provider_data, dict) else None
    return RequestProviderDataContext(
        provider_data=copied_provider_data,
        user=User(
            memory_config.default_vector_store_admin_principal,
            memory_config.default_vector_store_admin_attributes,
        ),
    )


def _memory_access_context(memory_config: MemoryConfig, use_admin_context: bool) -> AbstractContextManager[None]:
    if use_admin_context:
        return _memory_admin_context(memory_config)
    return nullcontext()


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


def _limit_memory_summary_messages(
    messages: list[OpenAIMessageParam],
    max_summary_messages: int,
) -> list[OpenAIMessageParam]:
    return messages[-max_summary_messages:]


def _format_memory_markdown(
    summary_text: str,
    messages: list[OpenAIMessageParam],
    owner_id: str,
    conversation_id: str,
    response_id: str,
    created_at: int,
    max_transcript_chars: int,
) -> str:
    transcript = _format_memory_transcript(messages, max_transcript_chars=max_transcript_chars)
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


def _format_memory_transcript(messages: list[OpenAIMessageParam], max_transcript_chars: int) -> str:
    lines: list[str] = []
    for message in messages:
        text = _message_content_to_text(message.content).strip()
        if text:
            lines.append(f"- {message.role}: {text}")
    return _truncate_text("\n".join(lines), max_transcript_chars)


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
    if _estimate_tokens_for_chars(used_chars) > max_context_tokens:
        return None

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
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_SUFFIX):
        return _TRUNCATION_SUFFIX.strip()
    return text[: max_chars - len(_TRUNCATION_SUFFIX)].rstrip() + _TRUNCATION_SUFFIX
