# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Security tests for path traversal, file ID validation, Content-Disposition
sanitization, and conversation ID format enforcement."""

import tempfile
from pathlib import Path

import pytest

from ogx.core.access_control.access_control import default_policy
from ogx.core.conversations.conversations import (
    ConversationServiceConfig,
    ConversationServiceImpl,
)
from ogx.core.datatypes import StackConfig
from ogx.core.storage.datatypes import (
    ServerStoresConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageConfig,
)
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.inline.files.localfs import (
    LocalfsFilesImpl,
    LocalfsFilesImplConfig,
)
from ogx.providers.utils.files.sanitize import (
    sanitize_content_disposition_filename,
)
from ogx_api import InvalidParameterError, OpenAIFilePurpose
from ogx_api.conversations import AddItemsRequest
from ogx_api.files.models import (
    RetrieveFileContentRequest,
    UploadFileRequest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockUploadFile:
    def __init__(self, content: bytes, filename: str):
        self.content = content
        self.filename = filename

    async def read(self):
        return self.content


@pytest.fixture
async def files_provider(tmp_path):
    storage_dir = tmp_path / "files"
    db_path = tmp_path / "files_metadata.db"

    backend_name = "sql_security_test"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=db_path.as_posix())})
    config = LocalfsFilesImplConfig(
        storage_dir=storage_dir.as_posix(),
        metadata_store=SqlStoreReference(backend=backend_name, table_name="files_metadata"),
    )

    provider = LocalfsFilesImpl(config, default_policy())
    await provider.initialize()
    yield provider


@pytest.fixture
async def conversation_service():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sec.db"
        storage = StorageConfig(
            backends={"sql_sec": SqliteSqlStoreConfig(db_path=str(db_path))},
            stores=ServerStoresConfig(
                conversations=SqlStoreReference(backend="sql_sec", table_name="conversations"),
                metadata=None,
                inference=None,
                prompts=None,
            ),
        )
        register_sqlstore_backends({"sql_sec": storage.backends["sql_sec"]})
        stack_config = StackConfig(distro_name="test", apis=[], providers={}, storage=storage)
        config = ConversationServiceConfig(config=stack_config, policy=[])
        svc = ConversationServiceImpl(config, {})
        await svc.initialize()
        yield svc


# ---------------------------------------------------------------------------
# TestFileIdValidation
# ---------------------------------------------------------------------------


class TestFileIdValidation:
    """Verify that _validate_file_id rejects malformed identifiers."""

    @pytest.mark.parametrize(
        "good_id",
        [
            "file-" + "a" * 32,
            "file-" + "0" * 32,
            "file-1",
            "file-843528265869",
            "file-" + "f" * 64,
        ],
    )
    def test_valid_file_ids_accepted(self, files_provider, good_id):
        files_provider._validate_file_id(good_id)

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../etc/passwd",
            "file-../../etc/passwd",
            "file-" + "A" * 32,
            "file-" + "g" * 32,
            "file-" + "a" * 65,
            "/etc/passwd",
            "file-\x00" + "a" * 31,
            "file-aa/../../etc/passwd",
            "file-",
            "",
            "not-a-file-id",
        ],
    )
    def test_malicious_file_ids_rejected(self, files_provider, bad_id):
        with pytest.raises(InvalidParameterError):
            files_provider._validate_file_id(bad_id)


# ---------------------------------------------------------------------------
# TestPathContainment
# ---------------------------------------------------------------------------


class TestPathContainment:
    """Verify that _validate_path_containment blocks escapes from storage_dir."""

    def test_valid_path_accepted(self, files_provider, tmp_path):
        storage_dir = Path(files_provider.config.storage_dir)
        valid = storage_dir / ("file-" + "a" * 32)
        result = files_provider._validate_path_containment(valid)
        assert result.is_relative_to(storage_dir.resolve())

    def test_traversal_via_dotdot_rejected(self, files_provider):
        storage_dir = Path(files_provider.config.storage_dir)
        malicious = storage_dir / ".." / "etc" / "passwd"
        with pytest.raises(InvalidParameterError):
            files_provider._validate_path_containment(malicious)

    def test_absolute_path_outside_storage_rejected(self, files_provider):
        with pytest.raises(InvalidParameterError):
            files_provider._validate_path_containment(Path("/etc/passwd"))

    def test_symlink_escape_rejected(self, files_provider, tmp_path):
        storage_dir = Path(files_provider.config.storage_dir)
        outside_file = tmp_path / "outside_secret.txt"
        outside_file.write_text("secret")
        symlink = storage_dir / "sneaky_link"
        symlink.symlink_to(outside_file)

        with pytest.raises(InvalidParameterError):
            files_provider._validate_path_containment(symlink)

    async def test_poisoned_db_path_rejected(self, files_provider):
        """Simulate a compromised DB row with a traversal path."""
        provider = files_provider
        assert provider.sql_store is not None

        file_id = provider._generate_file_id()
        poisoned_path = Path(provider.config.storage_dir) / ".." / "etc" / "passwd"

        await provider.sql_store.insert(
            "openai_files",
            {
                "id": file_id,
                "filename": "innocent.txt",
                "purpose": "assistants",
                "bytes": 10,
                "created_at": 1,
                "expires_at": 99999999999,
                "file_path": poisoned_path.as_posix(),
            },
        )

        with pytest.raises(InvalidParameterError):
            await provider._lookup_file_id(file_id)


# ---------------------------------------------------------------------------
# TestContentDispositionSanitization
# ---------------------------------------------------------------------------


class TestContentDispositionSanitization:
    """Verify filename sanitization for Content-Disposition headers."""

    def test_normal_filename_unchanged(self):
        assert sanitize_content_disposition_filename("report.pdf") == "report.pdf"

    def test_traversal_sequences_removed(self):
        result = sanitize_content_disposition_filename("../../etc/passwd")
        assert "/" not in result
        assert ".." not in result

    def test_null_bytes_stripped(self):
        result = sanitize_content_disposition_filename("file\x00name.txt")
        assert "\x00" not in result

    def test_hidden_file_gets_prefix(self):
        result = sanitize_content_disposition_filename(".htaccess")
        assert not result.startswith(".")

    def test_empty_filename_returns_download(self):
        assert sanitize_content_disposition_filename("") == "download"

    def test_backslashes_replaced(self):
        result = sanitize_content_disposition_filename("dir\\file.txt")
        assert "\\" not in result

    def test_slashes_replaced(self):
        result = sanitize_content_disposition_filename("dir/file.txt")
        assert "/" not in result

    async def test_upload_sanitizes_stored_filename(self, files_provider):
        """The returned OpenAIFileObject should already contain the sanitized name."""
        malicious_name = "../../etc/passwd"
        upload = MockUploadFile(b"test content", malicious_name)

        uploaded = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=upload,
        )

        assert ".." not in uploaded.filename
        assert "/" not in uploaded.filename

    async def test_content_disposition_header_uses_sanitized_name(self, files_provider):
        """End-to-end: upload a file with a malicious name, retrieve it, check the header."""
        malicious_name = "../../etc/passwd"
        upload = MockUploadFile(b"test content", malicious_name)

        uploaded = await files_provider.openai_upload_file(
            request=UploadFileRequest(purpose=OpenAIFilePurpose.ASSISTANTS),
            file=upload,
        )

        response = await files_provider.openai_retrieve_file_content(
            request=RetrieveFileContentRequest(file_id=uploaded.id),
        )

        header = response.headers["content-disposition"]
        assert ".." not in header
        assert "/" not in header


# ---------------------------------------------------------------------------
# TestConversationIdValidation
# ---------------------------------------------------------------------------


class TestConversationIdValidation:
    """Verify conversation ID regex enforcement."""

    VALID_CONV_ID = "conv_" + "a" * 48

    def test_valid_conversation_id_accepted(self, conversation_service):
        conversation_service._validate_conversation_id(self.VALID_CONV_ID)

    @pytest.mark.parametrize(
        "bad_id",
        [
            "conv_" + "a" * 47,
            "conv_" + "a" * 49,
            "conv_" + "G" * 48,
            "conv_../../etc/passwd" + "a" * 30,
            "../../../etc/passwd",
            "",
            "not_a_conv_id",
            "conv_" + "\x00" + "a" * 47,
        ],
    )
    def test_malicious_conversation_ids_rejected(self, conversation_service, bad_id):
        with pytest.raises(InvalidParameterError):
            conversation_service._validate_conversation_id(bad_id)

    async def test_add_items_rejects_bad_conversation_id(self, conversation_service):
        """_get_validated_conversation calls _validate_conversation_id."""
        bad_id = "../../../etc/shadow"
        with pytest.raises(InvalidParameterError):
            await conversation_service.add_items(
                bad_id,
                AddItemsRequest(
                    items=[
                        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
                    ]
                ),
            )
