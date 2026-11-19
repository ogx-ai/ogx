# Python SDK Reference

## Shared Types

```python
from ogx_client.models import (
    InterleavedContent,
    InterleavedContentItem,
    ParamType,
    SamplingParams,
    SystemMessage,
    ToolResponseMessage,
    URL,
    UserMessage,
)
```

## Tools

Types:

```python
from ogx_client.models import ListToolDefsResponse, ToolDef
```

## Inspect

Types:

```python
from ogx_client.models import HealthInfo, VersionInfo
```

Methods:

- <code title="get /v1/health">client.inspect.health() -> HealthInfo</code>
- <code title="get /v1/version">client.inspect.version() -> VersionInfo</code>

## Inference

Types:

```python
from ogx_client.models import RerankRequest, RerankResponse
```

Methods:

- <code title="post /v1alpha/inference/rerank">client.inference.rerank(rerank_request) -> RerankResponse</code>

## Embeddings

Types:

```python
from ogx_client.models import (
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
```

Methods:

- <code title="post /v1/embeddings">client.embeddings.create(open_ai_embeddings_request_with_extra_body) -> OpenAIEmbeddingsResponse</code>

## VectorIo

:::warning DEPRECATED API

**This API is deprecated and will be removed in a future version.**

Use the OpenAI-compatible Vector Stores API instead:

- Instead of `client.vector_io.insert()`, use `client.vector_stores.files.create()`
- Instead of `client.vector_io.query()`, use `client.vector_stores.search()`

See the [RAG documentation](/docs/building_applications/rag) for migration examples.

Related: [Issue #2981](https://github.com/ogx-ai/ogx/issues/2981)

:::

Types:

```python
from ogx_client.models import QueryChunksResponse
```

Methods:

- <code title="post /v1/vector-io/insert">client.vector_io.insert(insert_chunks_request) -> None</code>
- <code title="post /v1/vector-io/query">client.vector_io.query(query_chunks_request) -> QueryChunksResponse</code>

## Models

Types:

```python
from ogx_client.models import (
    GetModelV1ModelsModelIdGet200Response,
    ListModelsV1ModelsGet200Response,
    OpenAIModel,
)
```

Methods:

- <code title="get /v1/models/{model_id}">client.models.retrieve(model_id) -> GetModelV1ModelsModelIdGet200Response</code>
- <code title="get /v1/models">client.models.list() -> ListModelsV1ModelsGet200Response</code>

## Providers

Types:

```python
from ogx_client.models import ListProvidersResponse
```

Methods:

- <code title="get /v1/providers">client.providers.list() -> ListProvidersResponse</code>

## Routes

Types:

```python
from ogx_client.models import ListRoutesResponse
```

Methods:

- <code title="get /v1/inspect/routes">client.routes.list(api_filter=None) -> ListRoutesResponse</code>
