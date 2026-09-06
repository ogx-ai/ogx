# Python SDK Reference

## Shared Types

```python
from ogx_client.types import (
    AgentConfig,
    BatchCompletion,
    CompletionMessage,
    ContentDelta,
    Document,
    InterleavedContent,
    InterleavedContentItem,
    Message,
    ParamType,
    QueryConfig,
    QueryResult,
    ReturnType,
    SamplingParams,
    ScoringResult,
    SystemMessage,
    ToolCall,
    ToolParamDefinition,
    ToolResponseMessage,
    URL,
    UserMessage,
)
```

## Toolgroups

Types:

```python
from ogx_client.types import (
    ListToolGroupsResponse,
    ToolGroup,
    ToolgroupListResponse,
)
```

Methods:

- <code title="get /v1/toolgroups">client.toolgroups.list() -> ToolgroupListResponse</code>
- <code title="get /v1/toolgroups/{toolgroup_id}">client.toolgroups.get(toolgroup_id) -> ToolGroup</code>
- <code title="post /v1/toolgroups">client.toolgroups.register(\*\*params) -> None</code>
- <code title="delete /v1/toolgroups/{toolgroup_id}">client.toolgroups.unregister(toolgroup_id) -> None</code>

## Tools

Types:

```python
from ogx_client.types import ListToolsResponse, Tool, ToolListResponse
```

Methods:

- <code title="get /v1/tools">client.tools.list(\*\*params) -> ToolListResponse</code>
- <code title="get /v1/tools/{tool_name}">client.tools.get(tool_name) -> Tool</code>

## ToolRuntime

Types:

```python
from ogx_client.types import ToolDef, ToolInvocationResult
```

Methods:

- <code title="post /v1/tool-runtime/invoke">client.tool_runtime.invoke_tool(\*\*params) -> ToolInvocationResult</code>
- <code title="get /v1/tool-runtime/list-tools">client.tool_runtime.list_tools(\*\*params) -> JSONLDecoder[ToolDef]</code>

### RagTool

Methods:

- <code title="post /v1/tool-runtime/rag-tool/insert">client.tool_runtime.rag_tool.insert(\*\*params) -> None</code>
- <code title="post /v1/tool-runtime/rag-tool/query">client.tool_runtime.rag_tool.query(\*\*params) -> QueryResult</code>

## Agents

:::warning DEPRECATED API

**The Agents API is deprecated. Use the [Responses API](/docs/building_applications/responses_vs_agents) instead.**

The Responses API provides equivalent functionality with an OpenAI-compatible interface. New applications should use `client.responses.create()` rather than the agents workflow below.

:::

Types:

```python
from ogx_client.types import (
    InferenceStep,
    MemoryRetrievalStep,
    ToolExecutionStep,
    ToolResponse,
    AgentCreateResponse,
)
```

Methods:

- <code title="post /v1/agents">client.agents.create(\*\*params) -> AgentCreateResponse</code>
- <code title="delete /v1/agents/{agent_id}">client.agents.delete(agent_id) -> None</code>

### Session

Types:

```python
from ogx_client.types.agents import Session, SessionCreateResponse
```

Methods:

- <code title="post /v1/agents/{agent_id}/session">client.agents.session.create(agent_id, \*\*params) -> SessionCreateResponse</code>
- <code title="get /v1/agents/{agent_id}/session/{session_id}">client.agents.session.retrieve(session_id, \*, agent_id, \*\*params) -> Session</code>
- <code title="delete /v1/agents/{agent_id}/session/{session_id}">client.agents.session.delete(session_id, \*, agent_id) -> None</code>

### Steps

Types:

```python
from ogx_client.types.agents import StepRetrieveResponse
```

Methods:

- <code title="get /v1/agents/{agent_id}/session/{session_id}/turn/{turn_id}/step/{step_id}">client.agents.steps.retrieve(step_id, \*, agent_id, session_id, turn_id) -> StepRetrieveResponse</code>

### Turn

Types:

```python
from ogx_client.types.agents import Turn, TurnCreateResponse
```

Methods:

- <code title="post /v1/agents/{agent_id}/session/{session_id}/turn">client.agents.turn.create(session_id, \*, agent_id, \*\*params) -> TurnCreateResponse</code>
- <code title="get /v1/agents/{agent_id}/session/{session_id}/turn/{turn_id}">client.agents.turn.retrieve(turn_id, \*, agent_id, session_id) -> Turn</code>

## Datasets

Types:

```python
from ogx_client.types import (
    ListDatasetsResponse,
    DatasetRetrieveResponse,
    DatasetListResponse,
)
```

Methods:

- <code title="get /v1/datasets/{dataset_id}">client.datasets.retrieve(dataset_id) -> Optional[DatasetRetrieveResponse]</code>
- <code title="get /v1/datasets">client.datasets.list() -> DatasetListResponse</code>
- <code title="post /v1/datasets">client.datasets.register(\*\*params) -> None</code>
- <code title="delete /v1/datasets/{dataset_id}">client.datasets.unregister(dataset_id) -> None</code>

## Eval

Types:

```python
from ogx_client.types import EvaluateResponse, Job
```

Methods:

- <code title="post /v1/eval/tasks/{benchmark_id}/evaluations">client.eval.evaluate_rows(benchmark_id, \*\*params) -> EvaluateResponse</code>
- <code title="post /v1/eval/tasks/{benchmark_id}/jobs">client.eval.run_eval(benchmark_id, \*\*params) -> Job</code>

### Jobs

Types:

```python
from ogx_client.types.eval import JobStatusResponse
```

Methods:

- <code title="get /v1/eval/tasks/{benchmark_id}/jobs/{job_id}/result">client.eval.jobs.retrieve(job_id, \*, benchmark_id) -> EvaluateResponse</code>
- <code title="delete /v1/eval/tasks/{benchmark_id}/jobs/{job_id}">client.eval.jobs.cancel(job_id, \*, benchmark_id) -> None</code>
- <code title="get /v1/eval/tasks/{benchmark_id}/jobs/{job_id}">client.eval.jobs.status(job_id, \*, benchmark_id) -> Optional[JobStatusResponse]</code>

## Inspect

Types:

```python
from ogx_client.types import HealthInfo, ProviderInfo, RouteInfo, VersionInfo
```

Methods:

- <code title="get /v1/health">client.inspect.health() -> HealthInfo</code>
- <code title="get /v1/version">client.inspect.version() -> VersionInfo</code>

## Inference

Types:

```python
from ogx_client.types import (
    CompletionResponse,
    EmbeddingsResponse,
    TokenLogProbs,
    InferenceChatCompletionResponse,
    InferenceCompletionResponse,
)
```

Methods:

- <code title="post /v1/inference/embeddings">client.inference.embeddings(\*\*params) -> EmbeddingsResponse</code>

## VectorIo

:::warning DEPRECATED API

**This API is deprecated and will be removed in a future version.**

Use the OpenAI-compatible Vector Stores API instead:

- Instead of `client.vector_io.insert()`, use `client.vector_stores.files.create()` and `client.vector_stores.files.chunks.create()`
- Instead of `client.vector_io.query()`, use `client.vector_stores.search()`

See the [RAG documentation](/docs/building_applications/rag) for migration examples.

Related: [Issue #2981](https://github.com/ogx-ai/ogx/issues/2981)

:::

Types:

```python
from ogx_client.types import QueryChunksResponse
```

Methods:

- <code title="post /v1/vector-io/insert">client.vector_io.insert(\*\*params) -> None</code>
- <code title="post /v1/vector-io/query">client.vector_io.query(\*\*params) -> QueryChunksResponse</code>

## VectorDBs

:::warning DEPRECATED API

**This API is deprecated and will be removed in a future version.**

Use the OpenAI-compatible Vector Stores API instead:

- Instead of `client.vector_dbs.register()`, use `client.vector_stores.create()`
- Instead of `client.vector_dbs.list()`, use `client.vector_stores.list()`
- Instead of `client.vector_dbs.retrieve()`, use `client.vector_stores.retrieve()`
- Instead of `client.vector_dbs.unregister()`, use `client.vector_stores.delete()`

See the [RAG documentation](/docs/building_applications/rag) for migration examples.

Related: [Issue #2981](https://github.com/ogx-ai/ogx/issues/2981)

:::

Types:

```python
from ogx_client.types import (
    ListVectorDBsResponse,
    VectorDBRetrieveResponse,
    VectorDBListResponse,
    VectorDBRegisterResponse,
)
```

Methods:

- <code title="get /v1/vector-dbs/{vector_db_id}">client.vector_dbs.retrieve(vector_db_id) -> Optional[VectorDBRetrieveResponse]</code>
- <code title="get /v1/vector-dbs">client.vector_dbs.list() -> VectorDBListResponse</code>
- <code title="post /v1/vector-dbs">client.vector_dbs.register(\*\*params) -> VectorDBRegisterResponse</code>
- <code title="delete /v1/vector-dbs/{vector_db_id}">client.vector_dbs.unregister(vector_db_id) -> None</code>

## Models

Types:

```python
from ogx_client.types import ListModelsResponse, Model, ModelListResponse
```

Methods:

- <code title="get /v1/models/{model_id}">client.models.retrieve(model_id) -> Optional[Model]</code>
- <code title="get /v1/models">client.models.list() -> ModelListResponse</code>
- <code title="post /v1/models">client.models.register(\*\*params) -> Model</code>
- <code title="delete /v1/models/{model_id}">client.models.unregister(model_id) -> None</code>

## PostTraining

:::warning UNAVAILABLE API

**The Post Training API is not currently available in OGX.** There are no active providers implementing this API. The SDK types remain for forward compatibility but these endpoints are non-functional.

:::

Types:

```python
from ogx_client.types import ListPostTrainingJobsResponse, PostTrainingJob
```

Methods:

- <code title="post /v1/post-training/preference-optimize">client.post_training.preference_optimize(\*\*params) -> PostTrainingJob</code>
- <code title="post /v1/post-training/supervised-fine-tune">client.post_training.supervised_fine_tune(\*\*params) -> PostTrainingJob</code>

### Job

Types:

```python
from ogx_client.types.post_training import (
    JobListResponse,
    JobArtifactsResponse,
    JobStatusResponse,
)
```

Methods:

- <code title="get /v1/post-training/jobs">client.post_training.job.list() -> JobListResponse</code>
- <code title="get /v1/post-training/job/artifacts">client.post_training.job.artifacts(\*\*params) -> Optional[JobArtifactsResponse]</code>
- <code title="post /v1/post-training/job/cancel">client.post_training.job.cancel(\*\*params) -> None</code>
- <code title="get /v1/post-training/job/status">client.post_training.job.status(\*\*params) -> Optional[JobStatusResponse]</code>

## Providers

Types:

```python
from ogx_client.types import ListProvidersResponse, ProviderListResponse
```

Methods:

- <code title="get /v1/inspect/providers">client.providers.list() -> ProviderListResponse</code>

## Routes

Types:

```python
from ogx_client.types import ListRoutesResponse, RouteListResponse
```

Methods:

- <code title="get /v1/inspect/routes">client.routes.list() -> RouteListResponse</code>

## SyntheticDataGeneration

:::warning UNAVAILABLE API

**The Synthetic Data Generation API is not currently available in OGX.** There are no active providers implementing this API. The SDK types remain for forward compatibility but these endpoints are non-functional.

:::

Types:

```python
from ogx_client.types import SyntheticDataGenerationResponse
```

Methods:

- <code title="post /v1/synthetic-data-generation/generate">client.synthetic_data_generation.generate(\*\*params) -> SyntheticDataGenerationResponse</code>

## Datasetio

Types:

```python
from ogx_client.types import PaginatedRowsResult
```

Methods:

- <code title="post /v1/datasetio/rows">client.datasetio.append_rows(\*\*params) -> None</code>
- <code title="get /v1/datasetio/rows">client.datasetio.get_rows_paginated(\*\*params) -> PaginatedRowsResult</code>

## Scoring

Types:

```python
from ogx_client.types import ScoringScoreResponse, ScoringScoreBatchResponse
```

Methods:

- <code title="post /v1/scoring/score">client.scoring.score(\*\*params) -> ScoringScoreResponse</code>
- <code title="post /v1/scoring/score-batch">client.scoring.score_batch(\*\*params) -> ScoringScoreBatchResponse</code>

## ScoringFunctions

Types:

```python
from ogx_client.types import (
    ListScoringFunctionsResponse,
    ScoringFn,
    ScoringFunctionListResponse,
)
```

Methods:

- <code title="get /v1/scoring-functions/{scoring_fn_id}">client.scoring_functions.retrieve(scoring_fn_id) -> Optional[ScoringFn]</code>
- <code title="get /v1/scoring-functions">client.scoring_functions.list() -> ScoringFunctionListResponse</code>
- <code title="post /v1/scoring-functions">client.scoring_functions.register(\*\*params) -> None</code>

## Benchmarks

Types:

```python
from ogx_client.types import (
    Benchmark,
    ListBenchmarksResponse,
    BenchmarkListResponse,
)
```

Methods:

- <code title="get /v1/eval-tasks/{benchmark_id}">client.benchmarks.retrieve(benchmark_id) -> Optional[Benchmark]</code>
- <code title="get /v1/eval-tasks">client.benchmarks.list() -> BenchmarkListResponse</code>
- <code title="post /v1/eval-tasks">client.benchmarks.register(\*\*params) -> None</code>
