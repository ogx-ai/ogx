# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import UTC, datetime
from typing import Any

from fastapi import Response, UploadFile

from ogx.core.access_control.datatypes import Action
from ogx.core.datatypes import AccessRule
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import sqlstore_impl
from ogx.providers.utils.files.sanitize import sanitize_content_disposition_filename
from ogx_api import (
    DeleteFileRequest,
    ExpiresAfter,
    Files,
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
from openai import OpenAI

from .config import OpenAIFilesImplConfig


def _make_file_object(
    *,
    id: str,
    filename: str,
    purpose: str,
    bytes: int,
    created_at: int,
    expires_at: int,
    **kwargs: Any,
) -> OpenAIFileObject:
    """
    Construct an OpenAIFileObject and normalize expires_at.

    If expires_at is greater than the max we treat it as no-expiration and
    return None for expires_at.
    """
    obj = OpenAIFileObject(
        id=id,
        filename=filename,
        purpose=OpenAIFilePurpose(purpose),
        bytes=bytes,
        created_at=created_at,
        expires_at=expires_at,
        status="processed",
        status_details="",
    )

    if obj.expires_at is not None and obj.expires_at > (obj.created_at + ExpiresAfter.MAX):
        obj.expires_at = None  # type: ignore

    return obj


class OpenAIFilesImpl(Files):
    """OpenAI Files API implementation."""

    def __init__(self, config: OpenAIFilesImplConfig, policy: list[AccessRule]) -> None:
        self._config = config
        self.policy = policy
        self._client: OpenAI | None = None
        self._sql_store: AuthorizedSqlStore | None = None

    def _now(self) -> int:
        """Return current UTC timestamp as int seconds."""
        return int(datetime.now(UTC).timestamp())

    async def _get_file(
        self, file_id: str, return_expired: bool = False, action: Action = Action.READ
    ) -> dict[str, Any]:
        where: dict[str, str | dict] = {"id": file_id}
        if not return_expired:
            where["expires_at"] = {">": self._now()}
        if not (row := await self.sql_store.fetch_one("openai_files", where=where, action=action)):
            raise OpenAIFileObjectNotFoundError(file_id)
        return row

    async def _delete_file(self, file_id: str) -> None:
        """Delete a file from OpenAI and the database."""
        try:
            self.client.files.delete(file_id)
        except Exception as e:
            # If file doesn't exist on OpenAI side, just remove from metadata store
            if "not found" not in str(e).lower():
                raise RuntimeError(f"Failed to delete file from OpenAI: {e}") from e

        await self.sql_store.delete("openai_files", where={"id": file_id})

    async def _delete_if_expired(self, file_id: str) -> None:
        """If the file exists and is expired, delete it."""
        if row := await self._get_file(file_id, return_expired=True):
            if (expires_at := row.get("expires_at")) and expires_at <= self._now():
                await self._delete_file(file_id)

    async def initialize(self) -> None:
        self._client = OpenAI(api_key=self._config.api_key)

        self._sql_store = AuthorizedSqlStore(sqlstore_impl(self._config.metadata_store), self.policy)
        await self._sql_store.create_table(
            "openai_files",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "filename": ColumnType.STRING,
                "purpose": ColumnType.STRING,
                "bytes": ColumnType.INTEGER,
                "created_at": ColumnType.INTEGER,
                "expires_at": ColumnType.INTEGER,
            },
        )

    async def shutdown(self) -> None:
        pass

    @property
    def client(self) -> OpenAI:
        assert self._client is not None, "Provider not initialized"
        return self._client

    @property
    def sql_store(self) -> AuthorizedSqlStore:
        assert self._sql_store is not None, "Provider not initialized"
        return self._sql_store

    async def openai_upload_file(
        self,
        request: UploadFileRequest,
        file: UploadFile,
    ) -> OpenAIFileObject:
        purpose = request.purpose
        expires_after = request.expires_after

        filename = sanitize_content_disposition_filename(getattr(file, "filename", None) or "uploaded_file")
        content = await file.read()
        file_size = len(content)

        created_at = self._now()

        expires_at = created_at + ExpiresAfter.MAX * 42
        if purpose == OpenAIFilePurpose.BATCH:
            expires_at = created_at + ExpiresAfter.MAX

        if expires_after is not None:
            expires_at = created_at + expires_after.seconds

        try:
            from io import BytesIO

            file_obj = BytesIO(content)
            file_obj.name = filename

            response = self.client.files.create(
                file=file_obj,
                purpose=purpose.value,
            )

            file_id = response.id

            entry: dict[str, Any] = {
                "id": file_id,
                "filename": filename,
                "purpose": purpose.value,
                "bytes": file_size,
                "created_at": created_at,
                "expires_at": expires_at,
            }

            await self.sql_store.insert("openai_files", entry)

            return _make_file_object(**entry)

        except Exception as e:
            raise RuntimeError(f"Failed to upload file to OpenAI: {e}") from e

    async def openai_list_files(
        self,
        request: ListFilesRequest,
    ) -> ListOpenAIFileResponse:
        after = request.after
        limit = request.limit
        order = request.order
        purpose = request.purpose

        if not order:
            order = Order.desc

        where_conditions: dict[str, Any] = {"expires_at": {">": self._now()}}
        if purpose:
            where_conditions["purpose"] = purpose.value

        paginated_result = await self.sql_store.fetch_all(
            table="openai_files",
            where=where_conditions,
            order_by=[("created_at", order.value)],
            cursor=("id", after) if after else None,
            limit=limit,
        )

        files = [_make_file_object(**row) for row in paginated_result.data]

        return ListOpenAIFileResponse(
            data=files,
            has_more=paginated_result.has_more,
            first_id=files[0].id if files else "",
            last_id=files[-1].id if files else "",
        )

    async def openai_retrieve_file(self, request: RetrieveFileRequest) -> OpenAIFileObject:
        file_id = request.file_id
        await self._delete_if_expired(file_id)
        row = await self._get_file(file_id)
        return _make_file_object(**row)

    async def openai_delete_file(self, request: DeleteFileRequest) -> OpenAIFileDeleteResponse:
        file_id = request.file_id
        await self._delete_if_expired(file_id)
        _ = await self._get_file(file_id, action=Action.DELETE)
        await self._delete_file(file_id)
        return OpenAIFileDeleteResponse(id=file_id, deleted=True)

    async def openai_retrieve_file_content(self, request: RetrieveFileContentRequest) -> Response:
        file_id = request.file_id
        await self._delete_if_expired(file_id)

        row = await self._get_file(file_id)

        try:
            response = self.client.files.content(file_id)
            file_content = response.content

        except Exception as e:
            if "not found" in str(e).lower():
                await self._delete_file(file_id)
                raise OpenAIFileObjectNotFoundError(file_id) from e
            raise RuntimeError(f"Failed to download file from OpenAI: {e}") from e

        return Response(
            content=file_content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{sanitize_content_disposition_filename(row["filename"])}"'
            },
        )
