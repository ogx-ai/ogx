# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import boto3
import pytest
from moto import mock_aws

from ogx.core.storage.datatypes import SqliteSqlStoreConfig, SqlStoreReference
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.remote.files.s3 import S3FilesImplConfig, get_adapter_impl


class MockUploadFile:
    def __init__(self, content: bytes, filename: str, content_type: str = "text/plain"):
        self.content = content
        self.filename = filename
        self.content_type = content_type

    async def read(self):
        return self.content


@pytest.fixture
def sample_text_file():
    content = b"Hello, this is a test file for the S3 Files API!"
    return MockUploadFile(content, "sample_text_file-0.txt")


@pytest.fixture
def sample_text_file2():
    content = b"Hello, this is a second test file for the S3 Files API!"
    return MockUploadFile(content, "sample_text_file-1.txt")


@pytest.fixture
def s3_config(tmp_path):
    db_path = tmp_path / "s3_files_metadata.db"

    backend_name = f"sql_s3_{tmp_path.name}"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=db_path.as_posix())})
    return S3FilesImplConfig(
        bucket_name=f"test-bucket-{tmp_path.name}",
        region="not-a-region",
        auto_create_bucket=True,
        metadata_store=SqlStoreReference(backend=backend_name, table_name="s3_files_metadata"),
    )


@pytest.fixture
def s3_client():
    # we use `with mock_aws()` because @mock_aws decorator does not support
    # being a generator
    with mock_aws():
        # must yield or the mock will be reset before it is used
        yield boto3.client("s3")


@pytest.fixture
async def s3_provider(s3_config, s3_client):  # s3_client provides the moto mock, don't remove it
    provider = await get_adapter_impl(s3_config, {})
    yield provider
    await provider.shutdown()
