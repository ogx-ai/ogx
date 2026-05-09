# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Checkpointing, batching, logging, ID mapping utilities."""

from __future__ import annotations

import json
import logging  # allow-direct-logging
import time
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger("rag-benchmarks")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class Checkpoint:
    """JSON-based checkpoint for resumable operations."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self.save()

    def setdefault(self, key: str, value):
        if key not in self.data:
            self.data[key] = value
            self.save()
        return self.data[key]


# ---------------------------------------------------------------------------
# ID Mapping
# ---------------------------------------------------------------------------


class IDMapping:
    """Bidirectional mapping between corpus doc IDs and OpenAI file IDs."""

    def __init__(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint
        self._doc_to_file: dict[str, str] = checkpoint.setdefault("doc_to_file", {})
        self._file_to_doc: dict[str, str] = checkpoint.setdefault("file_to_doc", {})

    def add(self, doc_id: str, file_id: str) -> None:
        self._doc_to_file[doc_id] = file_id
        self._file_to_doc[file_id] = doc_id
        self.checkpoint.set("doc_to_file", self._doc_to_file)
        self.checkpoint.set("file_to_doc", self._file_to_doc)

    def file_id(self, doc_id: str) -> str | None:
        return self._doc_to_file.get(doc_id)

    def doc_id(self, file_id: str) -> str | None:
        return self._file_to_doc.get(file_id)

    @property
    def uploaded_doc_ids(self) -> set[str]:
        return set(self._doc_to_file.keys())

    def __len__(self) -> int:
        return len(self._doc_to_file)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def batched(iterable, n: int):
    """Yield successive n-sized chunks from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Retry with backoff
# ---------------------------------------------------------------------------


def _is_retryable(exc: Exception) -> bool:
    """Return False for client errors (4xx) that won't succeed on retry."""
    status = getattr(exc, "status_code", None)
    if status is not None and 400 <= status < 500:
        return False
    return True


def retry_with_backoff(fn, max_retries: int = 8, base_delay: float = 2.0):
    """Call fn() with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if not _is_retryable(e) or attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2**attempt), 120.0)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def progress_bar(iterable, desc: str, total: int | None = None, **kwargs):
    """Thin wrapper around tqdm."""
    return tqdm(iterable, desc=desc, total=total, **kwargs)
