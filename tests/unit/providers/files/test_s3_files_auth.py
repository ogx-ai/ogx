# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

import pytest

from ogx.core.datatypes import User
from ogx.providers.remote.files.s3.files import S3FilesImpl
from ogx_api import (
    DeleteFileRequest,
    ListFilesRequest,
    OpenAIFilePurpose,
    ResourceNotFoundError,
    RetrieveFileContentRequest,
    RetrieveFileRequest,
    UploadFileRequest,
)


async def test_listing_hides_other_users_file(s3_provider, sample_text_file):
    """Listing should not show files uploaded by other users."""
    user_a = User("user-a", {"roles": ["team-a"]})
    user_b = User("user-b", {"roles": ["team-b"]})

    with patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_a
        uploaded = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

    with patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_b
        listed = await s3_provider.openai_list_files(request=ListFilesRequest())
        assert all(f.id != uploaded.id for f in listed.data)


def _make_request_for_op(op_name: str, file_id: str):
    """Helper to create the appropriate request object for each operation."""
    if op_name == "retrieve":
        return RetrieveFileRequest(file_id=file_id)
    elif op_name == "content":
        return RetrieveFileContentRequest(file_id=file_id)
    elif op_name == "delete":
        return DeleteFileRequest(file_id=file_id)
    raise ValueError(f"Unknown op: {op_name}")


@pytest.mark.parametrize(
    "op,op_name",
    [
        (S3FilesImpl.openai_retrieve_file, "retrieve"),
        (S3FilesImpl.openai_retrieve_file_content, "content"),
        (S3FilesImpl.openai_delete_file, "delete"),
    ],
    ids=["retrieve", "content", "delete"],
)
async def test_cannot_access_other_user_file(s3_provider, sample_text_file, op, op_name):
    """Operations (metadata/content/delete) on another user's file should raise ResourceNotFoundError.

    `op` is an async callable (provider, request) -> awaits the requested operation.
    """
    user_a = User("user-a", {"roles": ["team-a"]})
    user_b = User("user-b", {"roles": ["team-b"]})

    with patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_a
        uploaded = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

    with patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_b
        with pytest.raises(ResourceNotFoundError):
            await op(s3_provider, _make_request_for_op(op_name, uploaded.id))


async def test_shared_role_allows_listing(s3_provider, sample_text_file):
    """Listing should show files uploaded by other users when roles are shared."""
    user_a = User("user-a", {"roles": ["shared-role"]})
    user_b = User("user-b", {"roles": ["shared-role"]})

    with patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_a
        uploaded = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

    with patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_b
        listed = await s3_provider.openai_list_files(request=ListFilesRequest())
        assert any(f.id == uploaded.id for f in listed.data)


@pytest.mark.parametrize(
    "op,op_name",
    [
        (S3FilesImpl.openai_retrieve_file, "retrieve"),
        (S3FilesImpl.openai_retrieve_file_content, "content"),
        (S3FilesImpl.openai_delete_file, "delete"),
    ],
    ids=["retrieve", "content", "delete"],
)
async def test_shared_role_allows_access(s3_provider, sample_text_file, op, op_name):
    """Operations (metadata/content/delete) on another user's file should succeed when users share a role.

    `op` is an async callable (provider, request) -> awaits the requested operation.
    """
    user_x = User("user-x", {"roles": ["shared-role"]})
    user_y = User("user-y", {"roles": ["shared-role"]})

    with patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_x
        uploaded = await s3_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=sample_text_file,
        )

    with patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_y
        await op(s3_provider, _make_request_for_op(op_name, uploaded.id))
