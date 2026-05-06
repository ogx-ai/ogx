# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from fastapi import Response, UploadFile

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

from ogx.core.access_control.datatypes import Action
from ogx.core.datatypes import AccessRule
from ogx.core.id_generation import generate_object_id
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import sqlstore_impl
from ogx.providers.utils.files.sanitize import sanitize_content_disposition_filename
from ogx_api import (
    ExpiresAfter,
    Files,
    ListOpenAIFileResponse,
    OpenAIFileDeleteResponse,
    OpenAIFileObject,
    OpenAIFileObjectNotFoundError,
    OpenAIFilePurpose,
    Order,
)
from ogx_api.files.models import (
    DeleteFileRequest,
    ListFilesRequest,
    RetrieveFileContentRequest,
    RetrieveFileRequest,
    UploadFileRequest,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

from .config import S3FilesImplConfig

# TODO: provider data for S3 credentials


def _create_s3_client(config: S3FilesImplConfig) -> "S3Client":
    try:
        s3_config = {
            "region_name": config.region,
        }

        # endpoint URL if specified (for MinIO, LocalStack, etc.)
        if config.endpoint_url:
            s3_config["endpoint_url"] = config.endpoint_url

        if config.aws_access_key_id and config.aws_secret_access_key:
            s3_config.update(
                {
                    "aws_access_key_id": config.aws_access_key_id.get_secret_value(),
                    "aws_secret_access_key": config.aws_secret_access_key.get_secret_value(),
                }
            )

        # Both cast and type:ignore are needed here:
        # - cast tells mypy the return type for downstream usage (S3Client vs generic client)
        # - type:ignore suppresses the call-overload error from boto3's complex overloaded signatures
        return cast("S3Client", boto3.client("s3", **s3_config))  # type: ignore[call-overload]

    except (BotoCoreError, NoCredentialsError) as e:
        raise RuntimeError(f"Failed to initialize S3 client: {e}") from e


async def _create_bucket_if_not_exists(client: "S3Client", config: S3FilesImplConfig) -> None:
    try:
        client.head_bucket(Bucket=config.bucket_name)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            if not config.auto_create_bucket:
                raise RuntimeError(
                    f"S3 bucket '{config.bucket_name}' does not exist. "
                    f"Either create the bucket manually or set 'auto_create_bucket: true' in your configuration."
                ) from e
            try:
                # For us-east-1, we can't specify LocationConstraint
                if config.region == "us-east-1":
                    client.create_bucket(Bucket=config.bucket_name)
                else:
                    client.create_bucket(
                        Bucket=config.bucket_name,
                        CreateBucketConfiguration=cast(Any, {"LocationConstraint": config.region}),
                    )
            except ClientError as create_error:
                raise RuntimeError(
                    f"Failed to create S3 bucket '{config.bucket_name}': {create_error}"
                ) from create_error
        elif error_code == "403":
            raise RuntimeError(f"Access denied to S3 bucket '{config.bucket_name}'") from e
        else:
            raise RuntimeError(f"Failed to access S3 bucket '{config.bucket_name}': {e}") from e


def _make_file_object(
    *,
    id: str,
    filename: str,
    purpose: str,
    bytes: int,
    created_at: int,
    expires_at: int,
    **kwargs: Any,  # here to ignore any additional fields, e.g. extra fields from AuthorizedSqlStore
) -> OpenAIFileObject:
    """
    Construct an OpenAIFileObject and normalize expires_at.

    If expires_at is greater than the max we treat it as no-expiration and
    return None for expires_at.

    The OpenAI spec says expires_at type is Integer, but the implementation
    will return None for no expiration.
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


class S3FilesImpl(Files):
    """S3-based implementation of the Files API."""

    def __init__(self, config: S3FilesImplConfig, policy: list[AccessRule]) -> None:
        self._config = config
        self.policy = policy
        self._client: S3Client | None = None
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
        """Delete a file from S3 and the database."""
        try:
            self.client.delete_object(
                Bucket=self._config.bucket_name,
                Key=file_id,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchKey":
                raise RuntimeError(f"Failed to delete file from S3: {e}") from e

        await self.sql_store.delete("openai_files", where={"id": file_id})

    async def _delete_if_expired(self, file_id: str) -> None:
        """If the file exists and is expired, delete it."""
        if row := await self._get_file(file_id, return_expired=True):
            if (expires_at := row.get("expires_at")) and expires_at <= self._now():
                await self._delete_file(file_id)

    async def initialize(self) -> None:
        self._client = _create_s3_client(self._config)
        await _create_bucket_if_not_exists(self._client, self._config)

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
                # TODO: add s3_etag field for integrity checking
            },
        )

    async def shutdown(self) -> None:
        pass

    @property
    def client(self) -> "S3Client":
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

        file_id = generate_object_id("file", lambda: f"file-{uuid.uuid4().hex}")

        filename = sanitize_content_disposition_filename(getattr(file, "filename", None) or "uploaded_file")

        created_at = self._now()

        # the default is no expiration.
        # to implement no expiration we set an expiration beyond the max.
        # we'll hide this fact from users when returning the file object.
        expires_at = created_at + ExpiresAfter.MAX * 42
        # the default for BATCH files is 30 days, which happens to be the expiration max.
        if purpose == OpenAIFilePurpose.BATCH:
            expires_at = created_at + ExpiresAfter.MAX

        if expires_after is not None:
            expires_at = created_at + expires_after.seconds

        content = await file.read()
        file_size = len(content)

        entry: dict[str, Any] = {
            "id": file_id,
            "filename": filename,
            "purpose": purpose.value,
            "bytes": file_size,
            "created_at": created_at,
            "expires_at": expires_at,
        }

        await self.sql_store.insert("openai_files", entry)

        try:
            self.client.put_object(
                Bucket=self._config.bucket_name,
                Key=file_id,
                Body=content,
                # TODO: enable server-side encryption
            )
        except ClientError as e:
            await self.sql_store.delete("openai_files", where={"id": file_id})

            raise RuntimeError(f"Failed to upload file to S3: {e}") from e

        return _make_file_object(**entry)

    async def openai_list_files(
        self,
        request: ListFilesRequest,
    ) -> ListOpenAIFileResponse:
        after = request.after
        limit = request.limit
        order = request.order
        purpose = request.purpose

        # this purely defensive. it should not happen because the router also default to Order.desc.
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
            # empty string or None? spec says str, ref impl returns str | None, we go with spec
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
        _ = await self._get_file(file_id, action=Action.DELETE)  # raises if not found
        await self._delete_file(file_id)
        return OpenAIFileDeleteResponse(id=file_id, deleted=True)

    async def openai_retrieve_file_content(self, request: RetrieveFileContentRequest) -> Response:
        file_id = request.file_id
        await self._delete_if_expired(file_id)

        row = await self._get_file(file_id)

        try:
            response = self.client.get_object(
                Bucket=self._config.bucket_name,
                Key=row["id"],
            )
            # TODO: can we stream this instead of loading it into memory
            content = response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                await self._delete_file(file_id)
                raise OpenAIFileObjectNotFoundError(file_id) from e
            raise RuntimeError(f"Failed to download file from S3: {e}") from e

        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{sanitize_content_disposition_filename(row["filename"])}"'
            },
        )
