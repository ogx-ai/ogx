# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Internal file search over vector stores, shared by model-facing providers.

File search is not a routed API: there is a single implementation, and its
backend variability lives in the vector_io providers. This module holds the
composition that model-facing providers need on top of
``VectorIO.openai_search_vector_store`` (which also serves
``POST /v1/vector_stores/{id}/search``): multi-store fan-out, result merging,
model-facing prompt formatting, and citation extraction.
"""

import asyncio
from typing import Any

from ogx.core.datatypes import VectorStoresConfig
from ogx.log import get_logger
from ogx_api import (
    InvalidParameterError,
    OpenAISearchVectorStoreRequest,
    TextContentItem,
    VectorIO,
    VectorStoreSearchResponse,
)
from ogx_api.vector_io import SearchRankingOptions as FileSearchRankingOptions

logger = get_logger(name=__name__, category="core::file_search")


async def search_files(
    vector_io_api: VectorIO,
    vector_store_ids: list[str],
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    max_num_results: int | None = None,
    ranking_options: FileSearchRankingOptions | None = None,
    search_mode: str,
) -> list[VectorStoreSearchResponse]:
    """Search all vector stores in parallel and return the merged results.

    A store that raises InvalidParameterError (the caller sent a parameter it
    cannot honour, e.g. ranking_options.hybrid_search on a provider that cannot
    apply the weights) re-raises: that is the client's to fix, so it should
    reach responses.create as an error instead of degrading to an empty,
    indistinguishable result. Genuine store failures degrade to no results
    from that store.
    """

    async def search_single_store(vector_store_id: str) -> list[VectorStoreSearchResponse]:
        try:
            search_response = await vector_io_api.openai_search_vector_store(
                vector_store_id=vector_store_id,
                request=OpenAISearchVectorStoreRequest(
                    query=query,
                    filters=filters,
                    # The request model requires an int; a client-supplied null takes the
                    # same default the tool model uses, instead of failing validation per store.
                    max_num_results=max_num_results if max_num_results is not None else 10,
                    ranking_options=ranking_options,
                    rewrite_query=False,
                    search_mode=search_mode,
                ),
            )
            return search_response.data
        except InvalidParameterError:
            raise
        except Exception as e:
            logger.warning("Failed to search vector store", vector_store_id=vector_store_id, error=str(e))
            return []

    all_results = await asyncio.gather(*(search_single_store(vid) for vid in vector_store_ids))

    merged: list[VectorStoreSearchResponse] = []
    for results in all_results:
        merged.extend(results)
    return merged


def build_content_items(
    query: str,
    search_results: list[VectorStoreSearchResponse],
    config: VectorStoresConfig,
) -> list[TextContentItem]:
    """Format merged search results into the model-facing context content items."""
    enable_annotations = config.annotation_prompt_params.enable_annotations

    # Get templates
    header_template = config.file_search_params.header_template
    footer_template = config.file_search_params.footer_template
    context_template = config.context_prompt_params.context_template

    # Get annotation templates (use defaults if annotations disabled)
    if enable_annotations:
        chunk_annotation_template = config.annotation_prompt_params.chunk_annotation_template
        annotation_instruction_template = config.annotation_prompt_params.annotation_instruction_template
    else:
        # Use defaults from VectorStoresConfig when annotations disabled
        default_config = VectorStoresConfig()
        chunk_annotation_template = default_config.annotation_prompt_params.chunk_annotation_template
        annotation_instruction_template = default_config.annotation_prompt_params.annotation_instruction_template

    content_items: list[TextContentItem] = []
    content_items.append(TextContentItem(text=header_template.format(num_chunks=len(search_results))))

    unique_files = set()
    for i, result_item in enumerate(search_results):
        chunk_text = result_item.content[0].text if result_item.content else ""
        # Get file_id from attributes if result_item.file_id is empty
        file_id = result_item.file_id or (result_item.attributes.get("document_id") if result_item.attributes else None)
        metadata_text = f"document_id: {file_id}, score: {result_item.score}"
        if result_item.attributes:
            metadata_text += f", attributes: {result_item.attributes}"

        text_content = chunk_annotation_template.format(
            index=i + 1, metadata_text=metadata_text, file_id=file_id, chunk_text=chunk_text
        )
        content_items.append(TextContentItem(text=text_content))
        unique_files.add(file_id)

    content_items.append(TextContentItem(text=footer_template))

    annotation_instruction = ""
    if unique_files:
        annotation_instruction = annotation_instruction_template

    content_items.append(
        TextContentItem(
            text=context_template.format(
                query=query, num_chunks=len(search_results), annotation_instruction=annotation_instruction
            )
        )
    )
    return content_items


def build_citation_files(
    search_results: list[VectorStoreSearchResponse],
) -> dict[str | float | bool | None, str | float | bool | None]:
    """Map file_id to filename in descending score order.

    Handles missing attributes for old versions. Iterating in descending score
    order means that when a model doesn't cite inline and
    extract_citations_from_text falls back to a single file, it picks the most
    relevant one: dict insertion order determines which file_id lands first.

    Keys and values are loosely typed because the attribute fallbacks accept
    whatever the provider stored in the chunk attributes.
    """
    citation_files: dict[str | float | bool | None, str | float | bool | None] = {}
    for result in sorted(search_results, key=lambda r: r.score, reverse=True):
        file_id: str | float | bool | None = result.file_id
        if not file_id and result.attributes:
            file_id = result.attributes.get("document_id")

        filename: str | float | bool | None = result.filename
        if not filename and result.attributes:
            filename = result.attributes.get("filename")
        if not filename:
            filename = "unknown"

        if file_id not in citation_files:
            citation_files[file_id] = filename
    return citation_files


def build_metadata(
    search_results: list[VectorStoreSearchResponse],
    citation_files: dict[str | float | bool | None, str | float | bool | None],
) -> dict[str, Any]:
    return {
        "document_ids": [r.file_id for r in search_results],
        "chunks": [r.content[0].text if r.content else "" for r in search_results],
        "scores": [r.score for r in search_results],
        "attributes": [r.attributes or {} for r in search_results],
        "citation_files": citation_files,
    }
