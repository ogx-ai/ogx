# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""End-to-end RAG evaluation via Responses API with file_search tool."""

from __future__ import annotations

import json
import logging  # allow-direct-logging
import tempfile
import time
from pathlib import Path

from openai import OpenAI

from .utils import IDMapping, progress_bar, retry_with_backoff

logger = logging.getLogger("rag-benchmarks")


def rag_query(
    client: OpenAI,
    model: str,
    query: str,
    vector_store_ids: list[str],
    mapping: IDMapping,
    max_num_results: int = 10,
    previous_response_id: str | None = None,
    search_mode: str | None = None,
    extra_body: dict | None = None,
) -> dict:
    """Execute a single RAG query via Responses API.

    Returns:
        {
            "answer": str,
            "response_id": str,
            "retrieved_docs": {doc_id: score},
            "retrieved_chunks": [{"doc_id": ..., "text": ..., "score": ...}],
        }
    """
    tool_config = {
        "type": "file_search",
        "vector_store_ids": vector_store_ids,
        "max_num_results": max_num_results,
    }
    if search_mode:
        tool_config["search_mode"] = search_mode

    kwargs: dict = {
        "model": model,
        "input": query,
        "tools": [tool_config],
        "include": ["file_search_call.results"],
    }
    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = retry_with_backoff(lambda: client.responses.create(**kwargs))

    # Extract answer text and retrieved chunks
    answer = ""
    retrieved_docs: dict[str, float] = {}
    retrieved_chunks: list[dict] = []

    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    answer += content.text
        elif item.type == "file_search_call":
            for result in getattr(item, "results", []) or []:
                file_id = result.file_id
                doc_id = mapping.doc_id(file_id) if mapping else file_id
                score = result.score
                text = result.text

                if doc_id and (doc_id not in retrieved_docs or score > retrieved_docs[doc_id]):
                    retrieved_docs[doc_id] = score
                retrieved_chunks.append(
                    {
                        "doc_id": doc_id or file_id,
                        "text": text,
                        "score": score,
                    }
                )

    return {
        "answer": answer.strip(),
        "response_id": response.id,
        "retrieved_docs": retrieved_docs,
        "retrieved_chunks": retrieved_chunks,
    }


def rag_query_batch(
    client: OpenAI,
    model: str,
    queries: dict[str, str],
    vector_store_ids: list[str],
    mapping: IDMapping,
    max_num_results: int = 10,
    search_mode: str | None = None,
    use_batch_api: bool = False,
    batch_id: str | None = None,
    extra_body: dict | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, dict]:
    """Run RAG queries for a batch of independent questions.

    Args:
        use_batch_api: If True, use OpenAI Batch API for 50% cost savings and
            higher throughput. Only works with OpenAI SaaS, not OGX.
        batch_id: If provided, resume polling an existing batch instead of submitting.

    Returns:
        {query_id: rag_query result dict}
    """
    if use_batch_api or batch_id:
        return _rag_query_batch_api(
            client=client,
            model=model,
            queries=queries,
            vector_store_ids=vector_store_ids,
            mapping=mapping,
            max_num_results=max_num_results,
            search_mode=search_mode,
            batch_id=batch_id,
        )

    # Load checkpoint if it exists
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    results = {}
    if checkpoint_path and checkpoint_path.exists():
        results = json.loads(checkpoint_path.read_text())
        logger.info(f"Resumed {len(results)} cached query results from {checkpoint_path}")

    completed = 0
    for qid, query_text in progress_bar(queries.items(), desc="RAG queries", total=len(queries)):
        if qid in results:
            continue
        try:
            result = rag_query(
                client=client,
                model=model,
                query=query_text,
                vector_store_ids=vector_store_ids,
                mapping=mapping,
                max_num_results=max_num_results,
                search_mode=search_mode,
                extra_body=extra_body,
            )
            results[qid] = result
        except Exception as e:
            logger.error(f"RAG query failed for {qid}: {e}")
            results[qid] = {"answer": "", "response_id": None, "retrieved_docs": {}, "retrieved_chunks": []}

        completed += 1
        if checkpoint_path and completed % 50 == 0:
            checkpoint_path.write_text(json.dumps(results))
            logger.info(f"Checkpointed {len(results)} results to {checkpoint_path}")

    # Final save
    if checkpoint_path:
        checkpoint_path.write_text(json.dumps(results))

    return results


def _build_batch_request(
    custom_id: str,
    model: str,
    query: str,
    vector_store_ids: list[str],
    max_num_results: int,
    search_mode: str | None,
) -> dict:
    """Build a single Batch API request line for a RAG query."""
    tool_config: dict = {
        "type": "file_search",
        "vector_store_ids": vector_store_ids,
        "max_num_results": max_num_results,
    }
    if search_mode:
        tool_config["search_mode"] = search_mode

    body: dict = {
        "model": model,
        "input": query,
        "tools": [tool_config],
        "include": ["file_search_call.results"],
    }

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def _parse_batch_response(response_body: dict, mapping: IDMapping) -> dict:
    """Parse a single Batch API response into our standard result format."""
    answer = ""
    retrieved_docs: dict[str, float] = {}
    retrieved_chunks: list[dict] = []

    for item in response_body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    answer += content.get("text", "")
        elif item.get("type") == "file_search_call":
            for result in item.get("results", []) or []:
                file_id = result.get("file_id", "")
                doc_id = mapping.doc_id(file_id) if mapping else file_id
                score = result.get("score", 0.0)
                text = result.get("text", "")

                if doc_id and (doc_id not in retrieved_docs or score > retrieved_docs[doc_id]):
                    retrieved_docs[doc_id] = score
                retrieved_chunks.append({"doc_id": doc_id or file_id, "text": text, "score": score})

    return {
        "answer": answer.strip(),
        "response_id": response_body.get("id", ""),
        "retrieved_docs": retrieved_docs,
        "retrieved_chunks": retrieved_chunks,
    }


def _rag_query_batch_api(
    client: OpenAI,
    model: str,
    queries: dict[str, str],
    vector_store_ids: list[str],
    mapping: IDMapping,
    max_num_results: int = 10,
    search_mode: str | None = None,
    poll_interval: float = 30.0,
    batch_id: str | None = None,
) -> dict[str, dict]:
    """Submit all queries via OpenAI Batch API and poll for results.

    If batch_id is provided, skip submission and resume polling an existing batch.
    """
    max_poll_errors = 10

    if batch_id:
        # Resume polling an existing batch (with retry on transient auth errors)
        logger.info(f"Resuming batch: {batch_id}")
        for attempt in range(max_poll_errors):
            try:
                batch = client.batches.retrieve(batch_id)
                break
            except Exception as e:
                logger.warning(f"Resume retrieve attempt {attempt + 1}/{max_poll_errors}: {e}")
                if attempt == max_poll_errors - 1:
                    raise
                time.sleep(poll_interval)
        logger.info(f"Batch {batch.id}: {batch.status}")
    else:
        # Step 1: Build JSONL input file
        logger.info(f"Building batch request for {len(queries)} queries...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            input_path = f.name
            for qid, query_text in queries.items():
                request = _build_batch_request(
                    custom_id=qid,
                    model=model,
                    query=query_text,
                    vector_store_ids=vector_store_ids,
                    max_num_results=max_num_results,
                    search_mode=search_mode,
                )
                f.write(json.dumps(request) + "\n")

        # Step 2: Upload the JSONL file
        logger.info("Uploading batch input file...")
        with open(input_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        logger.info(f"Uploaded batch input: {batch_input_file.id}")
        Path(input_path).unlink()

        # Step 3: Create the batch
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        logger.info(f"Created batch: {batch.id} (status: {batch.status})")

    # Step 4: Poll for completion (with retry on transient auth errors)
    consecutive_errors = 0
    while batch.status not in ("completed", "failed", "cancelled", "expired"):
        time.sleep(poll_interval)
        try:
            batch = client.batches.retrieve(batch.id)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            logger.warning(f"Poll error ({consecutive_errors}/{max_poll_errors}): {e}")
            if consecutive_errors >= max_poll_errors:
                raise RuntimeError(
                    f"Batch {batch.id} — polling failed {max_poll_errors} times. Resume with: --batch-id {batch.id}"
                ) from e
            continue
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else len(queries)
        failed = batch.request_counts.failed if batch.request_counts else 0
        logger.info(f"Batch {batch.id}: {batch.status} — {completed}/{total} completed, {failed} failed")

    if batch.status != "completed":
        logger.error(f"Batch {batch.id} ended with status: {batch.status}")
        # Return empty results
        return {
            qid: {"answer": "", "response_id": None, "retrieved_docs": {}, "retrieved_chunks": []} for qid in queries
        }

    # Step 5: Download and parse results (with retry on transient auth errors)
    logger.info("Batch complete. Downloading results...")
    output_file_id = batch.output_file_id
    for attempt in range(max_poll_errors):
        try:
            content = client.files.content(output_file_id).text
            break
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1}/{max_poll_errors}: {e}")
            if attempt == max_poll_errors - 1:
                raise
            time.sleep(poll_interval)

    results: dict[str, dict] = {}
    error_count = 0
    for line in content.strip().split("\n"):
        item = json.loads(line)
        qid = item["custom_id"]
        if item.get("error"):
            logger.error(f"Batch error for query {qid}: {item['error']}")
            results[qid] = {"answer": "", "response_id": None, "retrieved_docs": {}, "retrieved_chunks": []}
            error_count += 1
        else:
            response_body = item["response"]["body"]
            results[qid] = _parse_batch_response(response_body, mapping)

    if error_count:
        logger.warning(f"{error_count}/{len(queries)} queries failed in batch")

    logger.info(f"Batch results parsed: {len(results)} queries processed")
    return results


def rag_conversation(
    client: OpenAI,
    model: str,
    turns: list[dict],
    vector_store_ids: list[str],
    mapping: IDMapping,
    max_num_results: int = 10,
    search_mode: str | None = None,
    extra_body: dict | None = None,
) -> list[dict]:
    """Process a multi-turn conversation sequentially, threading via previous_response_id.

    Args:
        turns: [{"query_id": ..., "query": ...}, ...]

    Returns:
        List of rag_query result dicts, one per turn.
    """
    results = []
    prev_response_id = None

    for turn in turns:
        try:
            result = rag_query(
                client=client,
                model=model,
                query=turn["query"],
                vector_store_ids=vector_store_ids,
                mapping=mapping,
                max_num_results=max_num_results,
                previous_response_id=prev_response_id,
                search_mode=search_mode,
                extra_body=extra_body,
            )
            prev_response_id = result["response_id"]
        except Exception as e:
            logger.error(f"Conversation turn {turn.get('query_id')} failed: {e}")
            result = {"answer": "", "response_id": None, "retrieved_docs": {}, "retrieved_chunks": []}
        results.append(result)

    return results
