# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from io import BytesIO
from unittest.mock import patch

import pytest
import requests

from ogx.core.datatypes import User
from ogx_api import OpenAIFilePurpose

purpose = OpenAIFilePurpose.ASSISTANTS


@pytest.fixture()
def provider_type_is_openai(ogx_client):
    providers = [provider for provider in ogx_client.providers.list() if provider.api == "files"]
    assert len(providers) == 1, "Expected exactly one files provider"
    return providers[0].provider_type == "remote::openai"


# a fixture to skip all these tests if a files provider is not available
@pytest.fixture(autouse=True)
def skip_if_no_files_provider(ogx_client):
    if not [provider for provider in ogx_client.providers.list() if provider.api == "files"]:
        pytest.skip("No files providers found")


def test_openai_client_basic_operations(openai_client, provider_type_is_openai):
    """Test basic file operations through OpenAI client."""
    from openai import NotFoundError

    client = openai_client

    test_content = b"files test content"

    uploaded_file = None

    try:
        # Upload file using OpenAI client
        with BytesIO(test_content) as file_buffer:
            file_buffer.name = "openai_test.txt"
            uploaded_file = client.files.create(file=file_buffer, purpose=purpose)

        # Verify basic response structure
        assert uploaded_file.id.startswith("file-")
        assert hasattr(uploaded_file, "filename")
        assert uploaded_file.filename == "openai_test.txt"

        # List files
        files_list = client.files.list()
        file_ids = [f.id for f in files_list.data]
        assert uploaded_file.id in file_ids

        # Retrieve file info
        retrieved_file = client.files.retrieve(uploaded_file.id)
        assert retrieved_file.id == uploaded_file.id

        # Retrieve file content
        # OpenAI provider does not allow content retrieval with many `purpose` values
        if not provider_type_is_openai:
            content_response = client.files.content(uploaded_file.id)
            assert content_response.content == test_content

        # Delete file
        delete_response = client.files.delete(uploaded_file.id)
        assert delete_response.deleted is True

        # Retrieve file should fail
        with pytest.raises(NotFoundError):
            client.files.retrieve(uploaded_file.id)

        # File should not be found in listing
        files_list = client.files.list()
        file_ids = [f.id for f in files_list.data]
        assert uploaded_file.id not in file_ids

        # Double delete should fail
        with pytest.raises(NotFoundError):
            client.files.delete(uploaded_file.id)

    finally:
        # Cleanup in case of failure
        if uploaded_file is not None:
            try:
                client.files.delete(uploaded_file.id)
            except NotFoundError:
                pass  # ignore 404


@pytest.mark.xfail(message="expires_after not available on all providers")
def test_expires_after(openai_client):
    """Test uploading a file with expires_after parameter."""
    client = openai_client

    uploaded_file = None
    try:
        with BytesIO(b"expires_after test") as file_buffer:
            file_buffer.name = "expires_after.txt"
            uploaded_file = client.files.create(
                file=file_buffer,
                purpose=purpose,
                expires_after={"anchor": "created_at", "seconds": 4545},
            )

        assert uploaded_file.expires_at is not None
        assert uploaded_file.expires_at == uploaded_file.created_at + 4545

        listed = client.files.list()
        ids = [f.id for f in listed.data]
        assert uploaded_file.id in ids

        retrieved = client.files.retrieve(uploaded_file.id)
        assert retrieved.id == uploaded_file.id

    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(uploaded_file.id)
            except Exception:
                pass


@pytest.mark.xfail(message="expires_after not available on all providers")
def test_expires_after_requests(openai_client):
    """Upload a file using requests multipart/form-data and bracketed expires_after fields.

    This ensures clients that send form fields like `expires_after[anchor]` and
    `expires_after[seconds]` are handled by the server.
    """
    base_url = f"{openai_client.base_url}files"

    uploaded_id = None
    try:
        files = {"file": ("expires_after_with_requests.txt", BytesIO(b"expires_after via requests"))}
        data = {
            "purpose": str(purpose),
            "expires_after[anchor]": "created_at",
            "expires_after[seconds]": "4545",
        }

        session = requests.Session()
        request = requests.Request("POST", base_url, files=files, data=data)
        prepared = session.prepare_request(request)
        resp = session.send(prepared, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        assert result.get("id", "").startswith("file-")
        uploaded_id = result["id"]
        assert result.get("created_at") is not None
        assert result.get("expires_at") == result["created_at"] + 4545

        list_resp = requests.get(base_url, timeout=30)
        list_resp.raise_for_status()
        listed = list_resp.json()
        ids = [f["id"] for f in listed.get("data", [])]
        assert uploaded_id in ids

        retrieve_resp = requests.get(f"{base_url}/{uploaded_id}", timeout=30)
        retrieve_resp.raise_for_status()
        retrieved = retrieve_resp.json()
        assert retrieved["id"] == uploaded_id

    finally:
        if uploaded_id:
            try:
                requests.delete(f"{base_url}/{uploaded_id}", timeout=30)
            except Exception:
                pass


@pytest.mark.xfail(message="User isolation broken for current providers, must be fixed.")
@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
def test_files_authentication_isolation(mock_get_authenticated_user, ogx_client):
    """Test that users can only access their own files."""
    from llama_stack_client import NotFoundError

    client = ogx_client

    # Create two test users
    user1 = User("user1", {"roles": ["user"], "teams": ["team-a"]})
    user2 = User("user2", {"roles": ["user"], "teams": ["team-b"]})

    # User 1 uploads a file
    mock_get_authenticated_user.return_value = user1
    test_content_1 = b"User 1's private file content"

    with BytesIO(test_content_1) as file_buffer:
        file_buffer.name = "user1_file.txt"
        user1_file = client.files.create(file=file_buffer, purpose=purpose)

    # User 2 uploads a file
    mock_get_authenticated_user.return_value = user2
    test_content_2 = b"User 2's private file content"

    with BytesIO(test_content_2) as file_buffer:
        file_buffer.name = "user2_file.txt"
        user2_file = client.files.create(file=file_buffer, purpose=purpose)

    try:
        # User 1 can see their own file
        mock_get_authenticated_user.return_value = user1
        user1_files = client.files.list()
        user1_file_ids = [f.id for f in user1_files.data]
        assert user1_file.id in user1_file_ids
        assert user2_file.id not in user1_file_ids  # Cannot see user2's file

        # User 2 can see their own file
        mock_get_authenticated_user.return_value = user2
        user2_files = client.files.list()
        user2_file_ids = [f.id for f in user2_files.data]
        assert user2_file.id in user2_file_ids
        assert user1_file.id not in user2_file_ids  # Cannot see user1's file

        # User 1 can retrieve their own file
        mock_get_authenticated_user.return_value = user1
        retrieved_file = client.files.retrieve(user1_file.id)
        assert retrieved_file.id == user1_file.id

        # User 1 cannot retrieve user2's file
        mock_get_authenticated_user.return_value = user1
        with pytest.raises(NotFoundError, match="not found"):
            client.files.retrieve(user2_file.id)

        # User 1 can access their file content
        mock_get_authenticated_user.return_value = user1
        content_response = client.files.content(user1_file.id)
        if isinstance(content_response, str):
            content = bytes(content_response, "utf-8")
        else:
            content = content_response.content
        assert content == test_content_1

        # User 1 cannot access user2's file content
        mock_get_authenticated_user.return_value = user1
        with pytest.raises(NotFoundError, match="not found"):
            client.files.content(user2_file.id)

        # User 1 can delete their own file
        mock_get_authenticated_user.return_value = user1
        delete_response = client.files.delete(user1_file.id)
        assert delete_response.deleted is True

        # User 1 cannot delete user2's file
        mock_get_authenticated_user.return_value = user1
        with pytest.raises(NotFoundError, match="not found"):
            client.files.delete(user2_file.id)

        # User 2 can still access their file after user1's file is deleted
        mock_get_authenticated_user.return_value = user2
        retrieved_file = client.files.retrieve(user2_file.id)
        assert retrieved_file.id == user2_file.id

        # Cleanup user2's file
        mock_get_authenticated_user.return_value = user2
        client.files.delete(user2_file.id)

    except Exception as e:
        # Cleanup in case of failure
        try:
            mock_get_authenticated_user.return_value = user1
            client.files.delete(user1_file.id)
        except Exception:
            pass
        try:
            mock_get_authenticated_user.return_value = user2
            client.files.delete(user2_file.id)
        except Exception:
            pass
        raise e


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
def test_files_authentication_shared_attributes(mock_get_authenticated_user, ogx_client, provider_type_is_openai):
    """Test access control with users having identical attributes."""
    client = ogx_client

    # Create users with identical attributes (required for default policy)
    user_a = User("user-a", {"roles": ["user"], "teams": ["shared-team"]})
    user_b = User("user-b", {"roles": ["user"], "teams": ["shared-team"]})

    # User A uploads a file
    mock_get_authenticated_user.return_value = user_a
    test_content = b"Shared attributes file content"

    with BytesIO(test_content) as file_buffer:
        file_buffer.name = "shared_attributes_file.txt"
        shared_file = client.files.create(file=file_buffer, purpose=purpose)

    try:
        # User B with identical attributes can access the file
        mock_get_authenticated_user.return_value = user_b
        files_list = client.files.list()
        file_ids = [f.id for f in files_list.data]

        # User B should be able to see the file due to identical attributes
        assert shared_file.id in file_ids

        # User B can retrieve file info
        retrieved_file = client.files.retrieve(shared_file.id)
        assert retrieved_file.id == shared_file.id

        # User B can access file content
        if not provider_type_is_openai:
            content_response = client.files.content(shared_file.id)
            if isinstance(content_response, str):
                content = bytes(content_response, "utf-8")
            else:
                content = content_response.content
            assert content == test_content

        # Cleanup
        mock_get_authenticated_user.return_value = user_a
        client.files.delete(shared_file.id)

    except Exception as e:
        # Cleanup in case of failure
        try:
            mock_get_authenticated_user.return_value = user_a
            client.files.delete(shared_file.id)
        except Exception:
            pass
        try:
            mock_get_authenticated_user.return_value = user_b
            client.files.delete(shared_file.id)
        except Exception:
            pass
        raise e


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
def test_files_authentication_anonymous_access(mock_get_authenticated_user, ogx_client, provider_type_is_openai):
    client = ogx_client

    # Simulate anonymous user (no authentication)
    mock_get_authenticated_user.return_value = None

    test_content = b"Anonymous file content"

    with BytesIO(test_content) as file_buffer:
        file_buffer.name = "anonymous_file.txt"
        anonymous_file = client.files.create(file=file_buffer, purpose=purpose)

    try:
        # Anonymous user should be able to access their own uploaded file
        files_list = client.files.list()
        file_ids = [f.id for f in files_list.data]
        assert anonymous_file.id in file_ids

        # Can retrieve file info
        retrieved_file = client.files.retrieve(anonymous_file.id)
        assert retrieved_file.id == anonymous_file.id

        # Can access file content
        if not provider_type_is_openai:
            content_response = client.files.content(anonymous_file.id)
            if isinstance(content_response, str):
                content = bytes(content_response, "utf-8")
            else:
                content = content_response.content
            assert content == test_content

        # Can delete the file
        delete_response = client.files.delete(anonymous_file.id)
        assert delete_response.deleted is True

    except Exception as e:
        # Cleanup in case of failure
        try:
            client.files.delete(anonymous_file.id)
        except Exception:
            pass
        raise e
