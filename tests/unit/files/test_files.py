# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from ogx.core.access_control.access_control import default_policy
from ogx.core.storage.datatypes import SqliteSqlStoreConfig, SqlStoreReference
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.inline.files.localfs import (
    LocalfsFilesImpl,
    LocalfsFilesImplConfig,
)
from ogx_api import OpenAIFilePurpose, Order, ResourceNotFoundError
from ogx_api.files.models import (
    DeleteFileRequest,
    ListFilesRequest,
    OpenAIFileObject,
    RetrieveFileContentRequest,
    RetrieveFileRequest,
    UploadFileRequest,
)


class MockUploadFile:
    """Mock UploadFile for testing file uploads."""

    def __init__(self, content: bytes, filename: str, content_type: str = "text/plain"):
        self.content = content
        self.filename = filename
        self.content_type = content_type

    async def read(self):
        return self.content


@pytest.fixture
async def files_provider(tmp_path):
    """Create a files provider with temporary storage for testing."""
    storage_dir = tmp_path / "files"
    db_path = tmp_path / "files_metadata.db"

    backend_name = "sql_localfs_test"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=db_path.as_posix())})
    config = LocalfsFilesImplConfig(
        storage_dir=storage_dir.as_posix(),
        metadata_store=SqlStoreReference(backend=backend_name, table_name="files_metadata"),
    )

    provider = LocalfsFilesImpl(config, default_policy())
    await provider.initialize()
    yield provider


@pytest.fixture
def sample_text_file():
    """Sample text file for testing."""
    content = b"Hello, this is a test file for the OpenAI Files API!"
    return MockUploadFile(content, "test.txt", "text/plain")


@pytest.fixture
def sample_json_file():
    """Sample JSON file for testing."""
    content = b'{"message": "Hello, World!", "type": "test"}'
    return MockUploadFile(content, "data.json", "application/json")


@pytest.fixture
def large_file():
    """Large file for testing file size handling."""
    content = b"x" * 1024 * 1024  # 1MB file
    return MockUploadFile(content, "large_file.bin", "application/octet-stream")


class TestOpenAIFilesAPI:
    """Test suite for OpenAI Files API endpoints."""

    async def test_upload_file_success(self, files_provider, sample_text_file):
        """Test successful file upload."""
        # Upload file
        result = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
        )

        # Verify response
        assert result.id.startswith("file-")
        assert result.filename == "test.txt"
        assert result.purpose == OpenAIFilePurpose.ASSISTANTS
        assert result.bytes == len(sample_text_file.content)
        assert result.created_at > 0
        assert result.expires_at is not None and result.expires_at > result.created_at

    async def test_upload_different_purposes(self, files_provider, sample_text_file):
        """Test uploading files with different purposes."""
        purposes = list(OpenAIFilePurpose)

        uploaded_files = []
        for purpose in purposes:
            result = await files_provider.openai_upload_file(
                request=UploadFileRequest(purpose=purpose), file=sample_text_file
            )
            uploaded_files.append(result)
            assert result.purpose == purpose

    async def test_upload_different_file_types(self, files_provider, sample_text_file, sample_json_file, large_file):
        """Test uploading different types and sizes of files."""
        files_to_test = [
            (sample_text_file, "test.txt"),
            (sample_json_file, "data.json"),
            (large_file, "large_file.bin"),
        ]

        for file_obj, expected_filename in files_to_test:
            result = await files_provider.openai_upload_file(
                request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=file_obj
            )
            assert result.filename == expected_filename
            assert result.bytes == len(file_obj.content)

    async def test_list_files_empty(self, files_provider):
        """Test listing files when no files exist."""
        result = await files_provider.openai_list_files(request=ListFilesRequest())

        assert result.data == []
        assert result.has_more is False
        assert result.first_id == ""
        assert result.last_id == ""

    async def test_list_files_with_content(self, files_provider, sample_text_file, sample_json_file):
        """Test listing files when files exist."""
        # Upload multiple files
        file1 = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
        )
        file2 = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_json_file
        )

        # List files
        result = await files_provider.openai_list_files(request=ListFilesRequest())

        assert len(result.data) == 2
        file_ids = [f.id for f in result.data]
        assert file1.id in file_ids
        assert file2.id in file_ids

    async def test_list_files_with_purpose_filter(self, files_provider, sample_text_file):
        """Test listing files with purpose filtering."""
        # Upload file with specific purpose
        uploaded_file = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
        )

        # List files with matching purpose
        result = await files_provider.openai_list_files(request=ListFilesRequest(purpose=OpenAIFilePurpose.ASSISTANTS))
        assert len(result.data) == 1
        assert result.data[0].id == uploaded_file.id
        assert result.data[0].purpose == OpenAIFilePurpose.ASSISTANTS

    async def test_list_files_with_limit(self, files_provider, sample_text_file):
        """Test listing files with limit parameter."""
        # Upload multiple files
        for _ in range(5):
            await files_provider.openai_upload_file(
                request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
            )

        # List with limit
        result = await files_provider.openai_list_files(request=ListFilesRequest(limit=3))
        assert len(result.data) == 3

    async def test_list_files_with_order(self, files_provider, sample_text_file):
        """Test listing files with different order."""
        # Upload multiple files
        files = []
        for _ in range(3):
            file = await files_provider.openai_upload_file(
                request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
            )
            files.append(file)

        # Test descending order (default)
        result_desc = await files_provider.openai_list_files(request=ListFilesRequest(order=Order.desc))
        assert len(result_desc.data) == 3
        # Most recent should be first
        assert result_desc.data[0].created_at >= result_desc.data[1].created_at >= result_desc.data[2].created_at

        # Test ascending order
        result_asc = await files_provider.openai_list_files(request=ListFilesRequest(order=Order.asc))
        assert len(result_asc.data) == 3
        # Oldest should be first
        assert result_asc.data[0].created_at <= result_asc.data[1].created_at <= result_asc.data[2].created_at

    async def test_retrieve_file_success(self, files_provider, sample_text_file):
        """Test successful file retrieval."""
        # Upload file
        uploaded_file = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
        )

        # Retrieve file
        retrieved_file = await files_provider.openai_retrieve_file(
            request=RetrieveFileRequest(file_id=uploaded_file.id)
        )

        # Verify response
        assert retrieved_file.id == uploaded_file.id
        assert retrieved_file.filename == uploaded_file.filename
        assert retrieved_file.purpose == uploaded_file.purpose
        assert retrieved_file.bytes == uploaded_file.bytes
        assert retrieved_file.created_at == uploaded_file.created_at
        assert retrieved_file.expires_at == uploaded_file.expires_at

    async def test_retrieve_file_not_found(self, files_provider):
        """Test retrieving a non-existent file."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await files_provider.openai_retrieve_file(request=RetrieveFileRequest(file_id="file-" + "0" * 32))

    async def test_retrieve_file_content_success(self, files_provider, sample_text_file):
        """Test successful file content retrieval."""
        # Upload file
        uploaded_file = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
        )

        # Retrieve file content
        content = await files_provider.openai_retrieve_file_content(
            request=RetrieveFileContentRequest(file_id=uploaded_file.id)
        )

        # Verify content
        assert content.body == sample_text_file.content

    async def test_retrieve_file_content_not_found(self, files_provider):
        """Test retrieving content of a non-existent file."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await files_provider.openai_retrieve_file_content(
                request=RetrieveFileContentRequest(file_id="file-" + "0" * 32)
            )

    async def test_delete_file_success(self, files_provider, sample_text_file):
        """Test successful file deletion."""
        # Upload file
        uploaded_file = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
        )

        # Verify file exists
        await files_provider.openai_retrieve_file(request=RetrieveFileRequest(file_id=uploaded_file.id))

        # Delete file
        delete_response = await files_provider.openai_delete_file(request=DeleteFileRequest(file_id=uploaded_file.id))

        # Verify delete response
        assert delete_response.id == uploaded_file.id
        assert delete_response.deleted is True

        # Verify file no longer exists
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await files_provider.openai_retrieve_file(request=RetrieveFileRequest(file_id=uploaded_file.id))

    async def test_delete_file_not_found(self, files_provider):
        """Test deleting a non-existent file."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await files_provider.openai_delete_file(request=DeleteFileRequest(file_id="file-" + "0" * 32))

    async def test_file_persistence_across_operations(self, files_provider, sample_text_file):
        """Test that files persist correctly across multiple operations."""
        # Upload file
        uploaded_file = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
        )

        # Verify it appears in listing
        files_list = await files_provider.openai_list_files(request=ListFilesRequest())
        assert len(files_list.data) == 1
        assert files_list.data[0].id == uploaded_file.id

        # Retrieve file info
        retrieved_file = await files_provider.openai_retrieve_file(
            request=RetrieveFileRequest(file_id=uploaded_file.id)
        )
        assert retrieved_file.id == uploaded_file.id

        # Retrieve file content
        content = await files_provider.openai_retrieve_file_content(
            request=RetrieveFileContentRequest(file_id=uploaded_file.id)
        )
        assert content.body == sample_text_file.content

        # Delete file
        await files_provider.openai_delete_file(request=DeleteFileRequest(file_id=uploaded_file.id))

        # Verify it's gone from listing
        files_list = await files_provider.openai_list_files(request=ListFilesRequest())
        assert len(files_list.data) == 0

    async def test_multiple_files_operations(self, files_provider, sample_text_file, sample_json_file):
        """Test operations with multiple files."""
        # Upload multiple files
        file1 = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
        )
        file2 = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_json_file
        )

        # Verify both exist
        files_list = await files_provider.openai_list_files(request=ListFilesRequest())
        assert len(files_list.data) == 2

        # Delete one file
        await files_provider.openai_delete_file(request=DeleteFileRequest(file_id=file1.id))

        # Verify only one remains
        files_list = await files_provider.openai_list_files(request=ListFilesRequest())
        assert len(files_list.data) == 1
        assert files_list.data[0].id == file2.id

        # Verify the remaining file is still accessible
        content = await files_provider.openai_retrieve_file_content(
            request=RetrieveFileContentRequest(file_id=file2.id)
        )
        assert content.body == sample_json_file.content

    async def test_file_id_uniqueness(self, files_provider, sample_text_file):
        """Test that each uploaded file gets a unique ID."""
        file_ids = set()

        # Upload same file multiple times
        for _ in range(10):
            uploaded_file = await files_provider.openai_upload_file(
                request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
            )
            assert uploaded_file.id not in file_ids, f"Duplicate file ID: {uploaded_file.id}"
            file_ids.add(uploaded_file.id)
            assert uploaded_file.id.startswith("file-")

    async def test_file_no_filename_handling(self, files_provider):
        """Test handling files with no filename."""
        file_without_name = MockUploadFile(b"content", None)  # No filename

        uploaded_file = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=file_without_name
        )

        assert uploaded_file.filename == "uploaded_file"  # Default filename

    async def test_openaifileobject_schema_has_no_type_object(self):
        """Test that schemas with properties have redundant 'type: object' removed (matching OpenAI spec)."""
        from scripts.openapi_generator.schema_transforms import _remove_type_object_from_openai_schemas

        schema = {
            "components": {
                "schemas": {
                    "OpenAIFileObject": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "object": {"type": "string", "const": "file"},
                        },
                        "required": ["id", "object"],
                    },
                    "SchemaWithoutProperties": {
                        "type": "object",
                    },
                }
            }
        }
        _remove_type_object_from_openai_schemas(schema)
        assert "type" not in schema["components"]["schemas"]["OpenAIFileObject"]
        assert schema["components"]["schemas"]["SchemaWithoutProperties"]["type"] == "object"

    async def test_file_content_response_schema_is_string(self):
        """Test that /files/{file_id}/content response schema is type: string (not $ref to Response)."""
        from unittest.mock import AsyncMock

        from ogx_api.files.api import Files
        from ogx_api.files.fastapi_routes import create_router

        mock_impl = AsyncMock(spec=Files)
        router = create_router(mock_impl)

        # Build a minimal FastAPI app to get the OpenAPI schema
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        schema = app.openapi()

        content_path = schema["paths"]["/v1/files/{file_id}/content"]
        response_200 = content_path["get"]["responses"]["200"]
        json_schema = response_200["content"]["application/json"]["schema"]
        assert json_schema == {"type": "string"}, f"Expected type: string, got: {json_schema}"

    async def test_expires_at_can_be_none(self):
        """Test that expires_at can be None (it's optional per OpenAI spec)."""
        file_obj = OpenAIFileObject(
            id="file-abc123",
            bytes=100,
            created_at=1234567890,
            expires_at=None,
            filename="test.txt",
            purpose=OpenAIFilePurpose.ASSISTANTS,
            status="processed",
            status_details="",
        )
        assert file_obj.expires_at is None

    async def test_status_field_present(self):
        """Test that status and status_details are present on OpenAIFileObject."""
        file_obj = OpenAIFileObject(
            id="file-abc123",
            bytes=100,
            created_at=1234567890,
            filename="test.txt",
            purpose=OpenAIFilePurpose.ASSISTANTS,
            status="processed",
            status_details="",
        )
        assert file_obj.status == "processed"
        assert file_obj.status_details == ""

    async def test_status_can_be_set(self):
        """Test that status and status_details can be explicitly set."""
        file_obj = OpenAIFileObject(
            id="file-abc123",
            bytes=100,
            created_at=1234567890,
            filename="test.txt",
            purpose=OpenAIFilePurpose.ASSISTANTS,
            status="error",
            status_details="File validation failed",
        )
        assert file_obj.status == "error"
        assert file_obj.status_details == "File validation failed"

    async def test_uploaded_file_has_status(self, files_provider, sample_text_file):
        """Test that uploaded files include status in their response."""
        result = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
        )
        assert result.status == "processed"
        assert result.status_details == ""

    async def test_after_pagination_works(self, files_provider, sample_text_file):
        """Test that 'after' pagination works correctly."""
        # Upload multiple files to test pagination
        uploaded_files = []
        for _ in range(5):
            file = await files_provider.openai_upload_file(
                request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS), file=sample_text_file
            )
            uploaded_files.append(file)

        # Get first page without 'after' parameter
        first_page = await files_provider.openai_list_files(request=ListFilesRequest(limit=2, order=Order.desc))
        assert len(first_page.data) == 2
        assert first_page.has_more is True

        # Get second page using 'after' parameter
        second_page = await files_provider.openai_list_files(
            request=ListFilesRequest(after=first_page.data[-1].id, limit=2, order=Order.desc)
        )
        assert len(second_page.data) <= 2

        # Verify no overlap between pages
        first_page_ids = {f.id for f in first_page.data}
        second_page_ids = {f.id for f in second_page.data}
        assert first_page_ids.isdisjoint(second_page_ids)
