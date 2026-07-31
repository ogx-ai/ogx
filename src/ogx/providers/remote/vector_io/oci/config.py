# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import KVStoreReference
from ogx_api import json_schema_type


@json_schema_type
class OCI26aiVectorIOConfig(BaseModel):
    """Configuration for the Oracle 26ai vector I/O provider."""

    conn_str: str = Field(description="Connection string for the given 26ai Service")
    user: str = Field(description="Username name to connect to the service")
    password: str = Field(description="Password to connect to the service")
    tnsnames_loc: str = Field(description="Directory location of the tsnanames.ora file")
    ewallet_pem_loc: str = Field(description="Directory location of the ewallet.pem file")
    ewallet_password: str = Field(description="Password for the ewallet.pem file")
    persistence: KVStoreReference = Field(description="Config for KV store backend")
    consistency_level: str = Field(description="The consistency level of the OCI26ai server", default="Strong")
    vector_datatype: str = Field(description="Vector datatype for embeddings", default="FLOAT32")

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "conn_str": "${env.OCI26AI_CONNECTION_STRING}",
            "user": "${env.OCI26AI_USER}",
            "password": "${env.OCI26AI_PASSWORD}",
            "tnsnames_loc": "${env.OCI26AI_TNSNAMES_LOC}",
            "ewallet_pem_loc": "${env.OCI26AI_EWALLET_PEM_LOC}",
            "ewallet_password": "${env.OCI26AI_EWALLET_PWD}",
            "vector_datatype": "${env.OCI26AI_VECTOR_DATATYPE:=FLOAT32}",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::oci26ai",
            ).model_dump(exclude_none=True),
        }
