# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from html import escape
from typing import Any

from ogx.core.request_headers import get_authenticated_user
from ogx.log import get_logger
from ogx.providers.inline.responses.builtin.config import MemoryConfig
from ogx_api import OpenAIResponseInput, OpenAIResponseMessage, VectorIO, VectorStoreNotFoundError
from ogx_api.responses.models import MemoryToolConfig
from ogx_api.vector_io.models import OpenAISearchVectorStoreRequest, VectorStoreSearchResponse

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
        raise ValueError("memory owner is required")

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
        logger.debug("Skipping memory retrieval without request opt-in")
        return None
    if not request_memory.enabled:
        return None

    vector_store_id = (
        request_memory.vector_store_id
        if request_memory.vector_store_id is not None
        else memory_config.default_vector_store_id
    )
    if not vector_store_id:
        logger.debug("Skipping memory retrieval without vector store")
        return None

    owner_id = _resolve_owner_id(request_memory, metadata, safety_identifier)
    if not owner_id:
        logger.debug("Skipping memory retrieval without owner")
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
        logger.debug("Skipping memory retrieval without query text")
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


def _resolve_owner_id(
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


def _normalize_owner_id(owner_id: str | None) -> str | None:
    if owner_id is None:
        return None
    owner_id = owner_id.strip()
    return owner_id or None


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


def _estimate_tokens(text: str) -> int:
    return _estimate_tokens_for_chars(len(text))


def _estimate_tokens_for_chars(char_count: int) -> int:
    return max(1, char_count // APPROX_CHARS_PER_TOKEN)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= len(_TRUNCATION_SUFFIX):
        return _TRUNCATION_SUFFIX.strip()
    return text[: max_chars - len(_TRUNCATION_SUFFIX)].rstrip() + _TRUNCATION_SUFFIX
