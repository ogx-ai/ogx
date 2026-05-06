# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Retrieval-only evaluation via Vector Stores Search API."""

from __future__ import annotations

import logging  # allow-direct-logging

from openai import OpenAI

from .utils import IDMapping, progress_bar, retry_with_backoff

logger = logging.getLogger("rag-benchmarks")

MAX_QUERY_LENGTH = 4096


def search_queries(
    client: OpenAI,
    vector_store_id: str,
    queries: dict[str, str],
    mapping: IDMapping,
    max_num_results: int = 10,
    search_mode: str | None = None,
) -> dict[str, dict[str, float]]:
    """Run retrieval-only queries against a vector store.

    Args:
        client: OpenAI client.
        vector_store_id: Target vector store.
        queries: {query_id: query_text}
        mapping: File-ID-to-doc-ID mapping from ingestion.
        max_num_results: Top-k results to retrieve.
        search_mode: "vector", "hybrid", or "keyword" (OGX only).

    Returns:
        pytrec_eval format: {query_id: {doc_id: score}}
    """
    results: dict[str, dict[str, float]] = {}

    for qid, query_text in progress_bar(queries.items(), desc="Searching", total=len(queries)):
        if len(query_text) > MAX_QUERY_LENGTH:
            logger.debug(f"Truncating query {qid} from {len(query_text)} to {MAX_QUERY_LENGTH} chars")
            query_text = query_text[:MAX_QUERY_LENGTH]

        kwargs: dict = {
            "vector_store_id": vector_store_id,
            "query": query_text,
            "max_num_results": max_num_results,
        }
        if search_mode:
            kwargs["extra_body"] = {"search_mode": search_mode}

        try:
            response = retry_with_backoff(lambda kw=kwargs: client.vector_stores.search(**kw))
        except Exception as e:
            logger.error(f"Search failed for query {qid}: {e}")
            results[qid] = {}
            continue

        doc_scores: dict[str, float] = {}
        for item in response.data:
            file_id = item.file_id
            doc_id = mapping.doc_id(file_id)
            if doc_id is None:
                logger.debug(f"Unknown file_id {file_id} in search results")
                continue
            score = item.score
            # Keep highest score if a doc appears multiple times (multiple chunks)
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

        results[qid] = doc_scores

    return results
