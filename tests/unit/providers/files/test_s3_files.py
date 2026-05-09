# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

from ogx_api import (
    DeleteFileRequest,
    ListFilesRequest,
    OpenAIFilePurpose,
    ResourceNotFoundError,
    RetrieveFileContentRequest,
    RetrieveFileRequest,
    UploadFileRequest,
)


class TestS3FilesImpl:
    """Test suite for S3 Files implementation."""

    async def test_upload_file(self, s3_provider, sample_text_file, s3_client, s3_config):
        """Test successful file upload."""
        sample_text_file.filename = "test_upload_file"
        result = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

        assert result.filename == sample_text_file.filename
        assert result.purpose == OpenAIFilePurpose.ASSISTANTS
        assert result.bytes == len(sample_text_file.content)
        assert result.id.startswith("file-")

        # Verify file exists in S3 backend
        response = s3_client.head_object(Bucket=s3_config.bucket_name, Key=result.id)
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200

    async def test_list_files_empty(self, s3_provider):
        """Test listing files when no files exist."""
        result = await s3_provider.openai_list_files(request=ListFilesRequest())

        assert len(result.data) == 0
        assert not result.has_more
        assert result.first_id == ""
        assert result.last_id == ""

    async def test_retrieve_file(self, s3_provider, sample_text_file):
        """Test retrieving file metadata."""
        sample_text_file.filename = "test_retrieve_file"
        uploaded = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

        retrieved = await s3_provider.openai_retrieve_file(request=RetrieveFileRequest(file_id=uploaded.id))

        assert retrieved.id == uploaded.id
        assert retrieved.filename == uploaded.filename
        assert retrieved.purpose == uploaded.purpose
        assert retrieved.bytes == uploaded.bytes

    async def test_retrieve_file_content(self, s3_provider, sample_text_file):
        """Test retrieving file content."""
        sample_text_file.filename = "test_retrieve_file_content"
        uploaded = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

        response = await s3_provider.openai_retrieve_file_content(
            request=RetrieveFileContentRequest(file_id=uploaded.id)
        )

        assert response.body == sample_text_file.content
        assert response.headers["Content-Disposition"] == f'attachment; filename="{sample_text_file.filename}"'

    async def test_delete_file(self, s3_provider, sample_text_file, s3_config, s3_client):
        """Test deleting a file."""
        sample_text_file.filename = "test_delete_file"
        uploaded = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

        delete_response = await s3_provider.openai_delete_file(request=DeleteFileRequest(file_id=uploaded.id))

        assert delete_response.id == uploaded.id
        assert delete_response.deleted is True

        with pytest.raises(ResourceNotFoundError, match="not found"):
            await s3_provider.openai_retrieve_file(request=RetrieveFileRequest(file_id=uploaded.id))

        # Verify file is gone from S3 backend
        with pytest.raises(ClientError) as exc_info:
            s3_client.head_object(Bucket=s3_config.bucket_name, Key=uploaded.id)
        assert exc_info.value.response["Error"]["Code"] == "404"

    async def test_list_files(self, s3_provider, sample_text_file, sample_text_file2):
        """Test listing files after uploading some."""
        sample_text_file.filename = "test_list_files_with_content_file1"
        file1 = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

        sample_text_file2.filename = "test_list_files_with_content_file2"
        file2 = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.BATCH),
            file=sample_text_file2,
        )

        result = await s3_provider.openai_list_files(request=ListFilesRequest())

        assert len(result.data) == 2
        file_ids = {f.id for f in result.data}
        assert file1.id in file_ids
        assert file2.id in file_ids

    async def test_list_files_with_purpose_filter(self, s3_provider, sample_text_file, sample_text_file2):
        """Test listing files with purpose filter."""
        sample_text_file.filename = "test_list_files_with_purpose_filter_file1"
        file1 = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

        sample_text_file2.filename = "test_list_files_with_purpose_filter_file2"
        await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.BATCH),
            file=sample_text_file2,
        )

        result = await s3_provider.openai_list_files(request=ListFilesRequest(purpose=OpenAIFilePurpose.ASSISTANTS))

        assert len(result.data) == 1
        assert result.data[0].id == file1.id
        assert result.data[0].purpose == OpenAIFilePurpose.ASSISTANTS

    async def test_nonexistent_file_retrieval(self, s3_provider):
        """Test retrieving a non-existent file raises error."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await s3_provider.openai_retrieve_file(request=RetrieveFileRequest(file_id="file-nonexistent"))

    async def test_nonexistent_file_content_retrieval(self, s3_provider):
        """Test retrieving content of a non-existent file raises error."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await s3_provider.openai_retrieve_file_content(
                request=RetrieveFileContentRequest(file_id="file-nonexistent")
            )

    async def test_nonexistent_file_deletion(self, s3_provider):
        """Test deleting a non-existent file raises error."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await s3_provider.openai_delete_file(request=DeleteFileRequest(file_id="file-nonexistent"))

    async def test_upload_file_without_filename(self, s3_provider, sample_text_file):
        """Test uploading a file without a filename uses the fallback."""
        del sample_text_file.filename
        result = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

        assert result.purpose == OpenAIFilePurpose.ASSISTANTS
        assert result.bytes == len(sample_text_file.content)

        retrieved = await s3_provider.openai_retrieve_file(request=RetrieveFileRequest(file_id=result.id))
        assert retrieved.filename == result.filename

    async def test_file_operations_when_s3_object_deleted(self, s3_provider, sample_text_file, s3_config, s3_client):
        """Test file operations when S3 object is deleted but metadata exists (negative test)."""
        sample_text_file.filename = "test_orphaned_metadata"
        uploaded = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

        # Directly delete the S3 object from the backend
        s3_client.delete_object(Bucket=s3_config.bucket_name, Key=uploaded.id)

        with pytest.raises(ResourceNotFoundError, match="not found") as exc_info:
            await s3_provider.openai_retrieve_file_content(request=RetrieveFileContentRequest(file_id=uploaded.id))
        assert uploaded.id in str(exc_info).lower()

        listed_files = await s3_provider.openai_list_files(request=ListFilesRequest())
        assert uploaded.id not in [file.id for file in listed_files.data]

    async def test_upload_file_s3_put_object_failure(self, s3_provider, sample_text_file, s3_config, s3_client):
        """Test that put_object failure results in exception and no orphaned metadata."""
        sample_text_file.filename = "test_s3_put_object_failure"

        def failing_put_object(*args, **kwargs):
            raise ClientError(
                error_response={"Error": {"Code": "SolarRadiation", "Message": "Bloop"}}, operation_name="PutObject"
            )

        with patch.object(s3_provider.client, "put_object", side_effect=failing_put_object):
            with pytest.raises(RuntimeError, match="Failed to upload file to S3"):
                await s3_provider.openai_upload_file(
                    request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
                    file=sample_text_file,
                )

        files_list = await s3_provider.openai_list_files(request=ListFilesRequest())
        assert len(files_list.data) == 0, "No file metadata should remain after failed upload"

    @pytest.mark.parametrize("purpose", [p for p in OpenAIFilePurpose if p != OpenAIFilePurpose.BATCH])
    async def test_default_no_expiration(self, s3_provider, sample_text_file, purpose):
        """Test that by default files have no expiration."""
        sample_text_file.filename = "test_default_no_expiration"
        uploaded = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=purpose),
            file=sample_text_file,
        )
        assert uploaded.expires_at is None, "By default files should have no expiration"

    async def test_default_batch_expiration(self, s3_provider, sample_text_file):
        """Test that by default batch files have an expiration."""
        sample_text_file.filename = "test_default_batch_an_expiration"
        uploaded = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.BATCH),
            file=sample_text_file,
        )
        assert uploaded.expires_at is not None, "By default batch files should have an expiration"
        thirty_days_seconds = 30 * 24 * 3600
        assert uploaded.expires_at == uploaded.created_at + thirty_days_seconds, (
            "Batch default expiration should be 30 days"
        )

    async def test_expired_file_is_unavailable(self, s3_provider, sample_text_file, s3_config, s3_client):
        """Uploaded file that has expired should not be listed or retrievable/deletable."""
        with patch.object(s3_provider, "_now") as mock_now:  # control time
            two_hours = 2 * 60 * 60

            mock_now.return_value = 0

            from ogx_api import ExpiresAfter

            sample_text_file.filename = "test_expired_file"
            uploaded = await s3_provider.openai_upload_file(
                request=UploadFileRequest(
                    purpose=OpenAIFilePurpose.ASSISTANTS,
                    expires_after=ExpiresAfter(anchor="created_at", seconds=two_hours),
                ),
                file=sample_text_file,
            )

            mock_now.return_value = two_hours * 2  # fast forward 4 hours

            listed = await s3_provider.openai_list_files(request=ListFilesRequest())
            assert uploaded.id not in [f.id for f in listed.data]

            with pytest.raises(ResourceNotFoundError, match="not found"):
                await s3_provider.openai_retrieve_file(request=RetrieveFileRequest(file_id=uploaded.id))

            with pytest.raises(ResourceNotFoundError, match="not found"):
                await s3_provider.openai_retrieve_file_content(request=RetrieveFileContentRequest(file_id=uploaded.id))

            with pytest.raises(ResourceNotFoundError, match="not found"):
                await s3_provider.openai_delete_file(request=DeleteFileRequest(file_id=uploaded.id))

        with pytest.raises(ClientError) as exc_info:
            s3_client.head_object(Bucket=s3_config.bucket_name, Key=uploaded.id)
        assert exc_info.value.response["Error"]["Code"] == "404"

        with pytest.raises(ResourceNotFoundError, match="not found"):
            await s3_provider._get_file(uploaded.id, return_expired=True)

    async def test_unsupported_expires_after_anchor(self, s3_provider, sample_text_file):
        """Unsupported anchor value should raise ValueError."""
        from ogx_api import ExpiresAfter

        sample_text_file.filename = "test_unsupported_expires_after_anchor"

        with pytest.raises(ValueError, match="Input should be 'created_at'"):
            await s3_provider.openai_upload_file(
                file=sample_text_file,
                purpose=OpenAIFilePurpose.ASSISTANTS,
                expires_after=ExpiresAfter(anchor="now", seconds=3600),  # type: ignore
            )

    async def test_nonint_expires_after_seconds(self, s3_provider, sample_text_file):
        """Non-integer seconds in expires_after should raise ValueError."""
        from ogx_api import ExpiresAfter

        sample_text_file.filename = "test_nonint_expires_after_seconds"

        with pytest.raises(ValueError, match="should be a valid integer"):
            await s3_provider.openai_upload_file(
                file=sample_text_file,
                purpose=OpenAIFilePurpose.ASSISTANTS,
                expires_after=ExpiresAfter(anchor="created_at", seconds="many"),  # type: ignore
            )

    async def test_expires_after_seconds_out_of_bounds(self, s3_provider, sample_text_file):
        """Seconds outside allowed range should raise ValueError."""
        from ogx_api import ExpiresAfter

        with pytest.raises(ValueError, match="greater than or equal to 3600"):
            await s3_provider.openai_upload_file(
                file=sample_text_file,
                purpose=OpenAIFilePurpose.ASSISTANTS,
                expires_after=ExpiresAfter(anchor="created_at", seconds=3599),
            )

        with pytest.raises(ValueError, match="less than or equal to 2592000"):
            await s3_provider.openai_upload_file(
                file=sample_text_file,
                purpose=OpenAIFilePurpose.ASSISTANTS,
                expires_after=ExpiresAfter(anchor="created_at", seconds=2592001),
            )
