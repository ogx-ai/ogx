# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Local-filesystem files provider with defense-in-depth path security.

Security boundaries
-------------------
* **Indirect references** -- Clients address files via opaque IDs
  (``file-<1-64 hex chars>``), never raw paths.
* **ID format validation** -- ``_validate_file_id`` rejects any ID that does
  not match ``^file-[0-9a-f]{1,64}$``, blocking path separators, traversal
  sequences, null bytes, and any non-hex characters.
* **Path containment** -- ``_validate_path_containment`` resolves symlinks and
  ``..`` components, then verifies the result stays inside ``storage_dir``.
* **Filename sanitization** -- User-supplied filenames are sanitized at upload
  time before being stored in the database or returned in API responses.  The
  same sanitizer is applied again in the ``Content-Disposition`` header on
  download as a belt-and-suspenders measure.
"""

import re
import time
import uuid
from pathlib import Path

from fastapi import Response, UploadFile

from ogx.core.access_control.datatypes import Action
from ogx.core.datatypes import AccessRule
from ogx.core.id_generation import generate_object_id
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import sqlstore_impl
from ogx.log import get_logger
from ogx.providers.utils.files.sanitize import sanitize_content_disposition_filename
from ogx_api import (
    DeleteFileRequest,
    Files,
    InvalidParameterError,
    ListFilesRequest,
    ListOpenAIFileResponse,
    OpenAIFileDeleteResponse,
    OpenAIFileObject,
    OpenAIFileObjectNotFoundError,
    OpenAIFilePurpose,
    Order,
    RetrieveFileContentRequest,
    RetrieveFileRequest,
    UploadFileRequest,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

from .config import LocalfsFilesImplConfig

logger = get_logger(name=__name__, category="files")


class LocalfsFilesImpl(Files):
    """Files provider that stores uploaded files on the local filesystem."""

    def __init__(self, config: LocalfsFilesImplConfig, policy: list[AccessRule]) -> None:
        self.config = config
        self.policy = policy
        self.sql_store: AuthorizedSqlStore | None = None

    async def initialize(self) -> None:
        """Initialize the files provider by setting up storage directory and metadata database."""
        # Create storage directory if it doesn't exist
        storage_path = Path(self.config.storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize SQL store for metadata
        self.sql_store = AuthorizedSqlStore(sqlstore_impl(self.config.metadata_store), self.policy)
        await self.sql_store.create_table(
            "openai_files",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "filename": ColumnType.STRING,
                "purpose": ColumnType.STRING,
                "bytes": ColumnType.INTEGER,
                "created_at": ColumnType.INTEGER,
                "expires_at": ColumnType.INTEGER,
                "file_path": ColumnType.STRING,  # Path to actual file on disk
            },
        )

    async def shutdown(self) -> None:
        pass

    _FILE_ID_PATTERN = re.compile(r"^file-[0-9a-f]{1,64}$")

    def _generate_file_id(self) -> str:
        """Generate a unique file ID for OpenAI API."""
        return generate_object_id("file", lambda: f"file-{uuid.uuid4().hex}")

    def _validate_file_id(self, file_id: str) -> None:
        """Validate that file_id contains only safe characters (hex digits) after the ``file-`` prefix."""
        if not self._FILE_ID_PATTERN.fullmatch(file_id):
            raise InvalidParameterError(
                "file_id", file_id, "Must match format 'file-' followed by 1-64 hex characters."
            )

    def _validate_path_containment(self, file_path: Path) -> Path:
        """Canonicalize *file_path* and verify it resides inside storage_dir.

        Returns the resolved (absolute, symlink-free) path so callers operate
        on the canonical location.  Raises ``InvalidParameterError`` when the
        resolved path escapes the storage directory boundary.
        """
        resolved = file_path.resolve()
        storage_dir = Path(self.config.storage_dir).resolve()
        if not resolved.is_relative_to(storage_dir):
            raise InvalidParameterError(
                "file_path",
                file_path.name,
                "File path does not resolve to a valid storage location.",
            )
        return resolved

    def _get_file_path(self, file_id: str) -> Path:
        """Get the filesystem path for a file ID."""
        self._validate_file_id(file_id)
        path = Path(self.config.storage_dir) / file_id
        return self._validate_path_containment(path)

    async def _lookup_file_id(self, file_id: str, action: Action = Action.READ) -> tuple[OpenAIFileObject, Path]:
        """Look up a OpenAIFileObject and filesystem path from its ID."""
        self._validate_file_id(file_id)

        if not self.sql_store:
            raise RuntimeError("Files provider not initialized")

        row = await self.sql_store.fetch_one("openai_files", where={"id": file_id}, action=action)
        if not row:
            raise OpenAIFileObjectNotFoundError(file_id)

        file_path = Path(row.pop("file_path"))
        file_path = self._validate_path_containment(file_path)
        return OpenAIFileObject(**row, status="processed", status_details=""), file_path

    # OpenAI Files API Implementation
    async def openai_upload_file(
        self,
        request: UploadFileRequest,
        file: UploadFile,
    ) -> OpenAIFileObject:
        """Upload a file that can be used across various endpoints."""
        if not self.sql_store:
            raise RuntimeError("Files provider not initialized")

        purpose = request.purpose
        expires_after = request.expires_after

        if expires_after is not None:
            logger.warning(
                "File expiration is not supported by this provider, ignoring expires_after",
                expires_after=expires_after,
            )

        file_id = self._generate_file_id()
        file_path = self._get_file_path(file_id)
        sanitized_name = sanitize_content_disposition_filename(file.filename or "uploaded_file")

        content = await file.read()
        file_size = len(content)

        with open(file_path, "wb") as f:
            f.write(content)

        created_at = int(time.time())
        expires_at = created_at + self.config.ttl_secs

        await self.sql_store.insert(
            "openai_files",
            {
                "id": file_id,
                "filename": sanitized_name,
                "purpose": purpose.value,
                "bytes": file_size,
                "created_at": created_at,
                "expires_at": expires_at,
                "file_path": file_path.as_posix(),
            },
        )

        return OpenAIFileObject(
            id=file_id,
            filename=sanitized_name,
            purpose=purpose,
            bytes=file_size,
            created_at=created_at,
            expires_at=expires_at,
            status="processed",
            status_details="",
        )

    async def openai_list_files(
        self,
        request: ListFilesRequest,
    ) -> ListOpenAIFileResponse:
        """Returns a list of files that belong to the user's organization."""
        if not self.sql_store:
            raise RuntimeError("Files provider not initialized")

        after = request.after
        limit = request.limit
        order = request.order
        purpose = request.purpose

        if not order:
            order = Order.desc

        where_conditions = {}
        if purpose:
            where_conditions["purpose"] = purpose.value

        paginated_result = await self.sql_store.fetch_all(
            table="openai_files",
            where=where_conditions if where_conditions else None,
            order_by=[("created_at", order.value)],
            cursor=("id", after) if after else None,
            limit=limit,
        )

        files = [
            OpenAIFileObject(
                id=row["id"],
                filename=row["filename"],
                purpose=OpenAIFilePurpose(row["purpose"]),
                bytes=row["bytes"],
                created_at=row["created_at"],
                expires_at=row["expires_at"],
                status="processed",
                status_details="",
            )
            for row in paginated_result.data
        ]

        return ListOpenAIFileResponse(
            data=files,
            has_more=paginated_result.has_more,
            first_id=files[0].id if files else "",
            last_id=files[-1].id if files else "",
        )

    async def openai_retrieve_file(self, request: RetrieveFileRequest) -> OpenAIFileObject:
        """Returns information about a specific file."""
        file_obj, _ = await self._lookup_file_id(request.file_id)

        return file_obj

    async def openai_delete_file(self, request: DeleteFileRequest) -> OpenAIFileDeleteResponse:
        """Delete a file."""
        file_id = request.file_id
        # Delete physical file
        _, file_path = await self._lookup_file_id(file_id, action=Action.DELETE)
        if file_path.exists():
            file_path.unlink()

        # Delete metadata from database
        assert self.sql_store is not None, "Files provider not initialized"
        await self.sql_store.delete("openai_files", where={"id": file_id})

        return OpenAIFileDeleteResponse(
            id=file_id,
            deleted=True,
        )

    async def openai_retrieve_file_content(self, request: RetrieveFileContentRequest) -> Response:
        """Returns the contents of the specified file."""
        file_id = request.file_id
        # Read file content
        file_obj, file_path = await self._lookup_file_id(file_id)

        if not file_path.exists():
            logger.warning("File underlying path is missing, deleting metadata", file_id=file_id, file_path=file_path)
            await self.openai_delete_file(DeleteFileRequest(file_id=file_id))
            raise OpenAIFileObjectNotFoundError(file_id)

        # Return as binary response with appropriate content type
        return Response(
            content=file_path.read_bytes(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{sanitize_content_disposition_filename(file_obj.filename)}"'
            },
        )
