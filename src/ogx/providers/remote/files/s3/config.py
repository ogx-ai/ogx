# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator

from ogx.core.storage.datatypes import SqlStoreReference


class S3FilesImplConfig(BaseModel):
    """Configuration for S3-based files provider."""

    bucket_name: str = Field(description="S3 bucket name to store files")
    region: str = Field(default="us-east-1", description="AWS region where the bucket is located")
    aws_access_key_id: SecretStr | None = Field(
        default=None, description="AWS access key ID (optional if using IAM roles)"
    )
    aws_secret_access_key: SecretStr | None = Field(
        default=None, description="AWS secret access key (optional if using IAM roles)"
    )
    endpoint_url: str | None = Field(default=None, description="Custom S3 endpoint URL (for MinIO, LocalStack, etc.)")
    auto_create_bucket: bool = Field(
        default=False, description="Automatically create the S3 bucket if it doesn't exist"
    )
    key_prefix: str = Field(
        default="",
        description="Folder within the bucket to store files under, e.g. 'ogx/files'. Empty stores files at the bucket root",
    )
    metadata_store: SqlStoreReference = Field(description="SQL store configuration for file metadata")

    @field_validator("key_prefix")
    @classmethod
    def normalize_key_prefix(cls, v: str) -> str:
        # S3 has no directories; a "folder" is just a key prefix ending in "/". Accepting
        # "a/b", "/a/b" and "a/b/" as the same folder keeps the env var forgiving.
        segments = [segment for segment in v.split("/") if segment]
        return f"{'/'.join(segments)}/" if segments else ""

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "bucket_name": "${env.S3_BUCKET_NAME}",  # no default, buckets must be globally unique
            "region": "${env.AWS_REGION:=us-east-1}",
            "aws_access_key_id": "${env.AWS_ACCESS_KEY_ID:=}",
            "aws_secret_access_key": "${env.AWS_SECRET_ACCESS_KEY:=}",
            "endpoint_url": "${env.S3_ENDPOINT_URL:=}",
            "auto_create_bucket": "${env.S3_AUTO_CREATE_BUCKET:=false}",
            "key_prefix": "${env.S3_KEY_PREFIX:=}",
            "metadata_store": SqlStoreReference(
                backend="sql_default",
                table_name="s3_files_metadata",
            ).model_dump(exclude_none=True),
        }
