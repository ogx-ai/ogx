# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal

from pydantic import BaseModel

from ogx_api.resource import Resource, ResourceType


# Internal resource type for storing the vector store routing and other information
class VectorStore(Resource):
    """Vector database resource for storing and querying vector embeddings.

    :param type: Type of resource, always 'vector_store' for vector stores
    :param embedding_model: Name of the embedding model to use for vector generation
    :param embedding_dimension: Dimension of the embedding vectors
    """

    type: Literal[ResourceType.vector_store] = ResourceType.vector_store

    embedding_model: str
    embedding_dimension: int
    vector_store_name: str | None = None

    @property
    def vector_store_id(self) -> str:
        return self.identifier

    @property
    def provider_vector_store_id(self) -> str | None:
        return self.provider_resource_id


class VectorStoreInput(BaseModel):
    """Input parameters for creating or configuring a vector database.

    :param vector_store_id: Unique identifier for the vector store
    :param embedding_model: Name of the embedding model to use for vector generation
    :param embedding_dimension: Dimension of the embedding vectors
    :param provider_vector_store_id: (Optional) Provider-specific identifier for the vector store
    :param vector_store_name: (Optional) Human-readable name for the vector store
    """

    vector_store_id: str
    embedding_model: str
    embedding_dimension: int
    provider_id: str | None = None
    provider_vector_store_id: str | None = None
    vector_store_name: str | None = None
