# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from pathlib import Path

import pytest
import requests


@pytest.fixture(autouse=True)
def skip_if_no_file_processor_provider(ogx_client, require_server):
    """Skip tests if not running against a server or no file_processors provider is registered."""
    providers = [p for p in ogx_client.providers.list() if p.api == "file_processors"]
    if not providers:
        pytest.skip("No file_processors provider registered")


@pytest.fixture(scope="session")
def process_url(ogx_client):
    """URL for the file-processors/process endpoint."""
    return f"{ogx_client.base_url}/v1alpha/file-processors/process"


@pytest.fixture(scope="session")
def test_pdf_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]  # tests/integration/
        / "responses"
        / "fixtures"
        / "pdfs"
        / "ogx_and_models.pdf"
    )


@pytest.fixture(scope="session")
def test_pdf_content(test_pdf_path: Path) -> bytes:
    with open(test_pdf_path, "rb") as f:
        return f.read()


class TestFileProcessors:
    """Provider-agnostic integration tests for the file-processors/process endpoint."""

    def test_process_file_basic(self, process_url, test_pdf_content):
        """Test basic file processing without chunking."""
        resp = requests.post(
            process_url,
            files={"file": ("test.pdf", test_pdf_content, "application/pdf")},
            timeout=120,
        )
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code} {resp.text}"

        data = resp.json()
        assert "chunks" in data
        assert "metadata" in data
        assert len(data["chunks"]) == 1  # Single chunk without chunking strategy

        metadata = data["metadata"]
        assert "processing_time_ms" in metadata
        assert metadata["processing_time_ms"] >= 0
        assert "page_count" in metadata
        assert metadata["page_count"] > 0

        chunk = data["chunks"][0]
        assert chunk["content"] is not None
        assert len(chunk["content"].strip()) > 0

    def test_process_file_with_auto_chunking(self, process_url, test_pdf_content):
        """Test file processing with auto chunking strategy."""
        chunking_strategy = json.dumps({"type": "auto"})
        resp = requests.post(
            process_url,
            files={"file": ("test.pdf", test_pdf_content, "application/pdf")},
            data={"chunking_strategy": chunking_strategy},
            timeout=120,
        )
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code} {resp.text}"

        data = resp.json()
        assert len(data["chunks"]) >= 1

        chunk_ids = set()
        for chunk in data["chunks"]:
            assert len(chunk["content"].strip()) > 0
            assert chunk["chunk_id"] is not None
            assert chunk["chunk_id"] not in chunk_ids
            chunk_ids.add(chunk["chunk_id"])

            assert chunk["chunk_metadata"] is not None
            assert chunk["chunk_metadata"]["content_token_count"] > 0

            assert "document_id" in chunk["metadata"]
            assert chunk["metadata"]["filename"] == "test.pdf"

    def test_process_file_with_static_chunking(self, process_url, test_pdf_content):
        """Test file processing with static chunking strategy."""
        chunking_strategy = json.dumps(
            {
                "type": "static",
                "static": {
                    "max_chunk_size_tokens": 256,
                    "chunk_overlap_tokens": 25,
                },
            }
        )
        resp = requests.post(
            process_url,
            files={"file": ("test.pdf", test_pdf_content, "application/pdf")},
            data={"chunking_strategy": chunking_strategy},
            timeout=120,
        )
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code} {resp.text}"

        data = resp.json()
        assert len(data["chunks"]) > 1  # Should create multiple chunks

        chunk_ids = set()
        for chunk in data["chunks"]:
            assert len(chunk["content"].strip()) > 0
            assert chunk["chunk_id"] not in chunk_ids
            chunk_ids.add(chunk["chunk_id"])
            assert chunk["chunk_metadata"]["content_token_count"] > 0

    def test_chunk_id_uniqueness(self, process_url, test_pdf_content):
        """Test chunk IDs are unique across chunks."""
        chunking_strategy = json.dumps({"type": "auto"})
        resp = requests.post(
            process_url,
            files={"file": ("test.pdf", test_pdf_content, "application/pdf")},
            data={"chunking_strategy": chunking_strategy},
            timeout=120,
        )
        assert resp.status_code == 200

        data = resp.json()
        chunk_ids = [chunk["chunk_id"] for chunk in data["chunks"]]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_no_file_returns_error(self, process_url):
        """Test that omitting both file and file_id returns an error."""
        resp = requests.post(process_url, timeout=30)
        assert resp.status_code in (400, 422, 500)
