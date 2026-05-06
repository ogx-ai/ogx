# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile

from ogx.providers.inline.file_processor.pypdf import PyPDFFileProcessorConfig
from ogx.providers.inline.file_processor.pypdf.pypdf import PyPDFFileProcessor
from ogx_api.common.errors import OpenAIFileObjectNotFoundError
from ogx_api.vector_io import (
    VectorStoreChunkingStrategyAuto,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
)

# Minimal valid PDF for testing edge cases
MINIMAL_PDF_CONTENT = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000010 00000 n
0000000060 00000 n
0000000120 00000 n
0000000210 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
295
%%EOF"""

# Empty PDF with no text content for testing edge cases
EMPTY_PDF_CONTENT = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f
0000000010 00000 n
0000000060 00000 n
0000000120 00000 n
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
200
%%EOF"""


class TestPyPDFFileProcessor:
    """Integration tests for PyPDF file processor."""

    @pytest.fixture
    def config(self) -> PyPDFFileProcessorConfig:
        """Default configuration for testing."""
        return PyPDFFileProcessorConfig(
            extract_metadata=True, clean_text=True, default_chunk_size_tokens=512, default_chunk_overlap_tokens=50
        )

    @pytest.fixture
    def processor(self, config: PyPDFFileProcessorConfig) -> PyPDFFileProcessor:
        """PyPDF processor instance for testing."""
        return PyPDFFileProcessor(config, files_api=AsyncMock())

    @pytest.fixture
    def test_pdf_path(self) -> Path:
        """Path to the test PDF file."""
        return (
            Path(__file__).resolve().parents[1]  # tests/integration/
            / "responses"
            / "fixtures"
            / "pdfs"
            / "ogx_and_models.pdf"
        )

    @pytest.fixture
    def test_pdf_content(self, test_pdf_path: Path) -> bytes:
        """Content of the test PDF file."""
        with open(test_pdf_path, "rb") as f:
            return f.read()

    @pytest.fixture
    def upload_file(self, test_pdf_content: bytes) -> UploadFile:
        """Mock UploadFile for testing."""
        pdf_buffer = io.BytesIO(test_pdf_content)
        return UploadFile(file=pdf_buffer, filename="ogx_and_models.pdf")

    async def test_process_file_basic(self, processor: PyPDFFileProcessor, upload_file: UploadFile):
        """Test basic file processing without chunking."""
        upload_file.file.seek(0)  # Rewind stream before use
        response = await processor.process_file(file=upload_file, chunking_strategy=None)

        # Verify response structure
        assert response.chunks is not None
        assert response.metadata is not None
        assert len(response.chunks) == 1  # Single chunk without chunking strategy

        # Verify metadata
        metadata = response.metadata
        assert metadata["processor"] == "pypdf"
        assert "processing_time_ms" in metadata
        assert isinstance(metadata["processing_time_ms"], int)
        assert metadata["processing_time_ms"] >= 0
        assert "page_count" in metadata
        assert metadata["page_count"] > 0
        assert metadata["extraction_method"] == "pypdf"
        assert "file_size_bytes" in metadata

        # Verify chunk content and metadata
        chunk = response.chunks[0]
        assert chunk.content is not None
        assert len(chunk.content.strip()) > 0
        assert chunk.chunk_id is not None
        assert chunk.chunk_metadata is not None
        assert chunk.chunk_metadata.content_token_count > 0
        uuid.UUID(chunk.chunk_metadata.document_id)  # Should be a valid UUID

    async def test_process_file_with_auto_chunking(self, processor: PyPDFFileProcessor, upload_file: UploadFile):
        """Test file processing with auto chunking strategy."""
        upload_file.file.seek(0)  # Rewind stream before use
        chunking_strategy = VectorStoreChunkingStrategyAuto()
        response = await processor.process_file(file=upload_file, chunking_strategy=chunking_strategy)

        # Verify response structure
        assert response.chunks is not None
        # Should create at least one chunk - multiple chunks only if document is large enough
        assert len(response.chunks) >= 1

        # Collect chunk IDs to verify uniqueness
        chunk_ids: set[str] = set()

        # Verify chunks have content and proper metadata
        for chunk in response.chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.chunk_id is not None
            assert chunk.chunk_id not in chunk_ids  # Ensure uniqueness
            chunk_ids.add(chunk.chunk_id)

            # Verify chunk metadata
            assert chunk.chunk_metadata is not None
            assert chunk.chunk_metadata.content_token_count > 0
            uuid.UUID(chunk.chunk_metadata.document_id)  # Should be a valid UUID
            assert chunk.chunk_metadata.chunk_window is not None  # Should be set by make_overlapped_chunks

            # Verify chunk metadata dict
            assert "document_id" in chunk.metadata
            uuid.UUID(chunk.metadata["document_id"])  # Should be a valid UUID
            assert chunk.metadata["filename"] == "ogx_and_models.pdf"

    async def test_process_file_with_static_chunking(self, processor: PyPDFFileProcessor, upload_file: UploadFile):
        """Test file processing with static chunking strategy."""
        upload_file.file.seek(0)  # Rewind stream before use
        static_config = VectorStoreChunkingStrategyStaticConfig(max_chunk_size_tokens=256, chunk_overlap_tokens=25)
        chunking_strategy = VectorStoreChunkingStrategyStatic(static=static_config)
        response = await processor.process_file(file=upload_file, chunking_strategy=chunking_strategy)

        # Verify response structure
        assert response.chunks is not None
        assert len(response.chunks) > 1  # Should create multiple chunks

        # Collect chunk IDs to verify uniqueness
        chunk_ids: set[str] = set()

        # Verify chunk sizes are within expected range
        for chunk in response.chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.chunk_id is not None
            assert chunk.chunk_id not in chunk_ids  # Ensure uniqueness
            chunk_ids.add(chunk.chunk_id)

            # Token count should be <= max_chunk_size_tokens (with some tolerance for tokenizer differences)
            assert (
                chunk.chunk_metadata.content_token_count <= static_config.max_chunk_size_tokens + 50  # Allow tolerance
            )
            assert chunk.chunk_metadata.content_token_count > 0
            uuid.UUID(chunk.chunk_metadata.document_id)  # Should be a valid UUID

    async def test_metadata_extraction(self, processor: PyPDFFileProcessor, upload_file: UploadFile):
        """Test PDF metadata extraction."""
        upload_file.file.seek(0)  # Rewind stream before use
        response = await processor.process_file(file=upload_file, chunking_strategy=None)

        # Verify PDF metadata is extracted
        metadata = response.metadata
        assert "page_count" in metadata
        assert isinstance(metadata["page_count"], int)
        assert metadata["page_count"] > 0

        # Check document-level metadata in chunks
        chunk = response.chunks[0]
        assert "filename" in chunk.metadata
        assert chunk.metadata["filename"] == "ogx_and_models.pdf"
        uuid.UUID(chunk.metadata["document_id"])  # Should be a valid UUID

    async def test_text_cleaning(self):
        """Test text cleaning functionality."""
        config = PyPDFFileProcessorConfig(clean_text=True)
        processor = PyPDFFileProcessor(config, files_api=AsyncMock())

        # Test the text cleaning method directly
        raw_text = "  This  has   multiple   spaces\n\n\nand   extra\n\n  newlines  "
        cleaned_text = processor._clean_text(raw_text)

        expected = "This has multiple spaces\nand extra\nnewlines"
        assert cleaned_text == expected

    async def test_no_text_cleaning(self, upload_file: UploadFile):
        """Test processing without text cleaning."""
        upload_file.file.seek(0)  # Rewind stream before use
        config = PyPDFFileProcessorConfig(clean_text=False)
        processor = PyPDFFileProcessor(config, files_api=AsyncMock())

        response = await processor.process_file(file=upload_file, chunking_strategy=None)

        # Verify processing works without cleaning
        assert response.chunks is not None
        assert len(response.chunks) == 1
        assert len(response.chunks[0].content.strip()) > 0

    async def test_no_metadata_extraction(self, upload_file: UploadFile):
        """Test processing without metadata extraction."""
        upload_file.file.seek(0)  # Rewind stream before use
        config = PyPDFFileProcessorConfig(extract_metadata=False)
        processor = PyPDFFileProcessor(config, files_api=AsyncMock())

        response = await processor.process_file(file=upload_file, chunking_strategy=None)

        # Basic metadata should still be present
        metadata = response.metadata
        assert metadata["processor"] == "pypdf"
        assert "processing_time_ms" in metadata

        # PDF metadata extraction should be minimal - page_count will be 0 when metadata extraction is disabled
        assert "page_count" in metadata
        assert isinstance(metadata["page_count"], int)
        assert metadata["page_count"] == 0  # No metadata extraction means no page count

    async def test_input_validation(self, processor: PyPDFFileProcessor):
        """Test input validation."""
        # Test no file or file_id provided
        with pytest.raises(ValueError, match="Either file or file_id must be provided"):
            await processor.process_file()

        # Test both file and file_id provided
        upload_file = UploadFile(file=io.BytesIO(b"test"), filename="test.pdf")
        with pytest.raises(ValueError, match="Cannot provide both file and file_id"):
            await processor.process_file(file=upload_file, file_id="test_id")

    async def test_nonexistent_file_id_raises_error(self):
        """Test that a non-existent file_id raises a clear error."""
        mock_files_api = AsyncMock()
        mock_files_api.openai_retrieve_file.side_effect = OpenAIFileObjectNotFoundError("nonexistent_id")

        config = PyPDFFileProcessorConfig()
        processor = PyPDFFileProcessor(config, files_api=mock_files_api)

        with pytest.raises(OpenAIFileObjectNotFoundError, match="not found"):
            await processor.process_file(file_id="nonexistent_id")

    async def test_minimal_pdf_processing(self, processor: PyPDFFileProcessor):
        """Test processing a minimal PDF with no extractable text."""
        upload_file = UploadFile(file=io.BytesIO(MINIMAL_PDF_CONTENT), filename="minimal.pdf")

        response = await processor.process_file(file=upload_file, chunking_strategy=None)

        # Minimal PDFs often have no extractable text - should return empty chunks
        assert response.chunks is not None
        assert len(response.chunks) == 0
        assert response.metadata["processor"] == "pypdf"

    async def test_options_parameter_ignored(self, processor: PyPDFFileProcessor, upload_file: UploadFile):
        """Test that options parameter is properly ignored."""
        upload_file.file.seek(0)  # Rewind stream before use
        options = {"some_option": "some_value", "another": 123}

        response = await processor.process_file(file=upload_file, options=options, chunking_strategy=None)

        # Should process successfully despite options
        assert response.chunks is not None
        assert len(response.chunks) == 1

    async def test_invalid_pdf_content(self, processor: PyPDFFileProcessor):
        """Test handling of invalid PDF content."""
        invalid_content = b"This is not a PDF file content"
        upload_file = UploadFile(file=io.BytesIO(invalid_content), filename="invalid.pdf")

        # Should raise an exception for invalid PDF
        with pytest.raises(
            (ValueError, RuntimeError, OSError, Exception)
        ):  # PyPDF will raise various exceptions for invalid content including PdfStreamError
            await processor.process_file(file=upload_file, chunking_strategy=None)

    async def test_empty_pdf_handling(self, processor: PyPDFFileProcessor):
        """Test handling of empty or contentless PDF."""
        upload_file = UploadFile(file=io.BytesIO(EMPTY_PDF_CONTENT), filename="empty.pdf")

        response = await processor.process_file(file=upload_file, chunking_strategy=None)

        # Empty PDFs should return empty chunks but still have metadata
        assert response.chunks is not None
        assert len(response.chunks) == 0
        assert response.metadata["processor"] == "pypdf"

    async def test_document_id_generation(self, processor: PyPDFFileProcessor, upload_file: UploadFile):
        """Test document ID generation logic."""
        upload_file.file.seek(0)  # Rewind stream before use
        response = await processor.process_file(file=upload_file, chunking_strategy=None)

        chunk = response.chunks[0]
        # Document ID should be a generated UUID
        uuid.UUID(chunk.chunk_metadata.document_id)  # Should be a valid UUID
        uuid.UUID(chunk.metadata["document_id"])  # Should be a valid UUID
        assert chunk.metadata["filename"] == "ogx_and_models.pdf"

    async def test_chunk_id_uniqueness(self, processor: PyPDFFileProcessor, upload_file: UploadFile):
        """Test chunk ID uniqueness across chunks."""
        upload_file.file.seek(0)  # Rewind stream before use
        chunking_strategy = VectorStoreChunkingStrategyAuto()
        response = await processor.process_file(file=upload_file, chunking_strategy=chunking_strategy)

        # Verify chunk IDs are unique (hash-based from make_overlapped_chunks)
        chunk_ids = [chunk.chunk_id for chunk in response.chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

        # Verify each chunk has a valid ID
        for chunk in response.chunks:
            assert chunk.chunk_id is not None
            assert chunk.chunk_id != ""
            assert chunk.chunk_metadata.chunk_id == chunk.chunk_id

    async def test_tokenizer_consistency(self, processor: PyPDFFileProcessor, upload_file: UploadFile):
        """Test that tokenizer is consistently used for token counting."""
        upload_file.file.seek(0)  # Rewind stream before use
        response = await processor.process_file(file=upload_file, chunking_strategy=None)

        chunk = response.chunks[0]
        # Verify tokenizer name is set correctly
        assert chunk.chunk_metadata.chunk_tokenizer == "tiktoken:cl100k_base"
        assert chunk.chunk_metadata.content_token_count > 0

    async def test_processing_time_metadata(self, processor: PyPDFFileProcessor, upload_file: UploadFile):
        """Test processing time is captured in metadata."""
        upload_file.file.seek(0)  # Rewind stream before use
        response = await processor.process_file(file=upload_file, chunking_strategy=None)

        metadata = response.metadata
        assert "processing_time_ms" in metadata
        assert isinstance(metadata["processing_time_ms"], int)
        assert metadata["processing_time_ms"] >= 0


# Additional configuration-specific tests
class TestPyPDFFileProcessorConfig:
    """Tests for PyPDF file processor configuration."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = PyPDFFileProcessorConfig()

        assert config.extract_metadata is True
        assert config.clean_text is True
        assert config.default_chunk_size_tokens >= 100
        assert config.default_chunk_overlap_tokens >= 0

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = PyPDFFileProcessorConfig(
            default_chunk_size_tokens=500,
            default_chunk_overlap_tokens=100,
            extract_metadata=False,
            clean_text=False,
        )
        assert config.default_chunk_size_tokens == 500
        assert config.default_chunk_overlap_tokens == 100
        assert config.extract_metadata is False
        assert config.clean_text is False
