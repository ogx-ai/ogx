# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile

from ogx.providers.inline.file_processor.pypdf.config import PyPDFFileProcessorConfig
from ogx.providers.inline.file_processor.pypdf.pypdf import PyPDFFileProcessor
from ogx_api.vector_io import VectorStoreChunkingStrategyAuto


@pytest.mark.parametrize(
    "word_count, expected_windows",
    [
        (300, ["0-300"]),
        (500, ["0-500"]),
        (1000, ["0-800", "400-1000"]),
        (1500, ["0-800", "400-1200", "800-1500"]),
    ],
)
async def test_auto_chunking_has_no_redundant_tail_chunk(word_count: int, expected_windows: list[str]) -> None:
    processor = PyPDFFileProcessor(PyPDFFileProcessorConfig(), MagicMock())
    assert processor.config.default_chunk_size_tokens == 800
    assert processor.config.default_chunk_overlap_tokens == 400

    text = " ".join(["hello"] * word_count)
    file = UploadFile(filename="notes.txt", file=io.BytesIO(text.encode()))

    result = await processor.process_file(file=file, chunking_strategy=VectorStoreChunkingStrategyAuto())

    assert [c.chunk_metadata.chunk_window for c in result.chunks] == expected_windows
