# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.log import get_logger
from ogx.providers.inline.responses.builtin.config import MemoryConfig
from ogx_api import OpenAIResponseInput, OpenAIResponseMessage, VectorIO
from ogx_api.responses.models import MemoryToolConfig
from ogx_api.vector_io.models import OpenAISearchVectorStoreRequest, VectorStoreSearchResponse

logger = get_logger(name=__name__, category="openai_responses::memory")

_APPROX_CHARS_PER_TOKEN = 4
_TRUNCATION_SUFFIX = "\n[truncated]"


def extract_memory_query(input: str | list[OpenAIResponseInput]) -> str:
    """Extract the latest user text from the current request input."""
    if isinstance(input, str):
        return input

    for item in reversed(input):
        if not isinstance(item, OpenAIResponseMessage) or item.role != "user":
            continue

        text_segments = _extract_text_from_message(item)
        if text_segments:
            return "\n".join(text_segments)

    return "Current user turn"


def build_memory_filters(
    memory_config: MemoryConfig,
    owner_id: str,
    request_filters: dict[str, Any] | None,
) -> dict[str, Any]:
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
    enabled = request_memory.enabled if request_memory is not None else memory_config.default_enabled
    if not enabled:
        return None

    vector_store_id = (
        request_memory.vector_store_id
        if request_memory is not None and request_memory.vector_store_id is not None
        else memory_config.default_vector_store_id
    )
    if not vector_store_id:
        logger.debug("Skipping memory retrieval without vector store")
        return None

    owner_id = _resolve_owner_id(request_memory, metadata, safety_identifier)
    if not owner_id:
        logger.debug("Skipping memory retrieval without owner")
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
    except Exception as exc:
        logger.warning(
            "Failed to retrieve memory context",
            vector_store_id=vector_store_id,
            owner_id=owner_id,
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
    if request_memory is not None and request_memory.owner_id:
        return request_memory.owner_id
    if safety_identifier:
        return safety_identifier
    if metadata:
        return metadata.get("owner_id") or metadata.get("user_id")
    return None


def _extract_text_from_message(message: OpenAIResponseMessage) -> list[str]:
    if isinstance(message.content, str):
        return [message.content]

    text_segments: list[str] = []
    for content_item in message.content:
        text = getattr(content_item, "text", None)
        if isinstance(text, str) and text:
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

    for result in results:
        snippet = _format_memory_result(len(snippets) + 1, result)
        candidate = "\n".join([opening, *snippets, snippet, closing])
        if _estimate_tokens(candidate) <= max_context_tokens:
            snippets.append(snippet)
            continue

        if not snippets:
            remaining_chars = max_context_tokens * _APPROX_CHARS_PER_TOKEN - len(opening) - len(closing) - 2
            snippets.append(_truncate_text(snippet, remaining_chars))
        break

    if not snippets:
        return None

    return "\n".join([opening, *snippets, closing])


def _format_memory_result(index: int, result: VectorStoreSearchResponse) -> str:
    attributes = result.attributes or {}
    created_at = attributes.get("created_at", "")
    text = "\n".join(content.text for content in result.content if content.text)
    return f'<memory index="{index}" file_id="{result.file_id}" created_at="{created_at}">\n{text}\n</memory>'


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= len(_TRUNCATION_SUFFIX):
        return _TRUNCATION_SUFFIX.strip()
    return text[: max_chars - len(_TRUNCATION_SUFFIX)].rstrip() + _TRUNCATION_SUFFIX
