# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Callers that build their own uploads instead of receiving a multipart-parsed one, run against the S3 provider."""

import asyncio
import io
import json
from unittest.mock import AsyncMock, MagicMock

from fastapi import UploadFile

from ogx.core.library_client import AsyncOGXAsLibraryClient
from ogx.core.storage.datatypes import SqliteSqlStoreConfig, SqlStoreReference
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import _sqlstore_impl, register_sqlstore_backends
from ogx.providers.inline.batches.reference.batches import ReferenceBatchesImpl
from ogx.providers.inline.batches.reference.config import ReferenceBatchesImplConfig
from ogx.providers.utils.files.response import response_body_bytes
from ogx_api import Api, RetrieveFileContentRequest, UploadFileRequest
from ogx_api.batches.models import CreateBatchRequest, RetrieveBatchRequest
from ogx_api.files.models import OpenAIFileUploadPurpose


async def test_library_client_upload_is_stored_in_s3(s3_provider, s3_config, s3_client, monkeypatch):
    class _Stack:
        def __init__(self, config, custom_provider_registry=None):
            self.impls = {Api.files: s3_provider}

        async def initialize(self):
            pass

        async def shutdown(self):
            pass

    monkeypatch.setattr("ogx.core.library_client.Stack", _Stack)
    client = AsyncOGXAsLibraryClient("ci-tests")
    await client.initialize()

    uploaded = await client.files.create(file=("notes.txt", b"hello from the library client"), purpose="assistants")

    assert uploaded.bytes == len(b"hello from the library client")
    stored = s3_client.get_object(Bucket=s3_config.bucket_name, Key=uploaded.id)
    assert stored["Body"].read() == b"hello from the library client"


async def test_batch_output_file_is_stored_in_s3(s3_provider, tmp_path):
    backend = f"sql_batches_{tmp_path.name}"
    register_sqlstore_backends({backend: SqliteSqlStoreConfig(db_path=(tmp_path / "batches.db").as_posix())})
    config = ReferenceBatchesImplConfig(sqlstore=SqlStoreReference(backend=backend, table_name="batches"))
    inference = AsyncMock()
    inference.openai_embeddings.return_value = MagicMock(model_dump_json=MagicMock(return_value="{}"))
    sql_store = AuthorizedSqlStore(await _sqlstore_impl(config.sqlstore), policy=[])
    batches = ReferenceBatchesImpl(config, inference, s3_provider, AsyncMock(), sql_store)
    await batches.initialize()

    request_line = {
        "custom_id": "req-1",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {"model": "test-model", "input": "hello"},
    }
    input_file = await s3_provider.openai_upload_file(
        request=UploadFileRequest(purpose=OpenAIFileUploadPurpose.BATCH),
        file=UploadFile(file=io.BytesIO(json.dumps(request_line).encode()), filename="input.jsonl"),
    )
    batch = await batches.create_batch(
        CreateBatchRequest(input_file_id=input_file.id, endpoint="/v1/embeddings", completion_window="24h")
    )
    await asyncio.wait_for(batches._processing_tasks[batch.id], timeout=5)

    completed = await batches.retrieve_batch(RetrieveBatchRequest(batch_id=batch.id))
    assert completed.status == "completed"
    assert completed.output_file_id is not None
    output = await s3_provider.openai_retrieve_file_content(
        RetrieveFileContentRequest(file_id=completed.output_file_id)
    )
    lines = (await response_body_bytes(output)).splitlines()
    assert [json.loads(line)["custom_id"] for line in lines] == ["req-1"]

    await batches.shutdown()
