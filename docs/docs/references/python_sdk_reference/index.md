# Python SDK Reference

## Shared Types

```python
from llama_stack_client.types import (
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
    SafetyViolation,
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
from llama_stack_client.types import (
    ListToolGroupsResponse,
    ToolGroup,
    ToolgroupListResponse,
)
```

Methods:

- <code title="get /v1/toolgroups">client.toolgroups.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/toolgroups.py">list</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/toolgroup_list_response.py">ToolgroupListResponse</a></code>
- <code title="get /v1/toolgroups/{toolgroup_id}">client.toolgroups.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/toolgroups.py">get</a>(toolgroup_id) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/tool_group.py">ToolGroup</a></code>
- <code title="post /v1/toolgroups">client.toolgroups.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/toolgroups.py">register</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/toolgroup_register_params.py">params</a>) -> None</code>
- <code title="delete /v1/toolgroups/{toolgroup_id}">client.toolgroups.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/toolgroups.py">unregister</a>(toolgroup_id) -> None</code>

## Tools

Types:

```python
from llama_stack_client.types import ListToolsResponse, Tool, ToolListResponse
```

Methods:

- <code title="get /v1/tools">client.tools.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/tools.py">list</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/tool_list_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/tool_list_response.py">ToolListResponse</a></code>
- <code title="get /v1/tools/{tool_name}">client.tools.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/tools.py">get</a>(tool_name) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/tool.py">Tool</a></code>

## ToolRuntime

Types:

```python
from llama_stack_client.types import ToolDef, ToolInvocationResult
```

Methods:

- <code title="post /v1/tool-runtime/invoke">client.tool_runtime.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/tool_runtime/tool_runtime.py">invoke_tool</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/tool_runtime_invoke_tool_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/tool_invocation_result.py">ToolInvocationResult</a></code>
- <code title="get /v1/tool-runtime/list-tools">client.tool_runtime.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/tool_runtime/tool_runtime.py">list_tools</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/tool_runtime_list_tools_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/tool_def.py">JSONLDecoder[ToolDef]</a></code>

### RagTool

Methods:

- <code title="post /v1/tool-runtime/rag-tool/insert">client.tool_runtime.rag_tool.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/tool_runtime/rag_tool.py">insert</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/tool_runtime/rag_tool_insert_params.py">params</a>) -> None</code>
- <code title="post /v1/tool-runtime/rag-tool/query">client.tool_runtime.rag_tool.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/tool_runtime/rag_tool.py">query</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/tool_runtime/rag_tool_query_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/shared/query_result.py">QueryResult</a></code>

## Agents

:::warning DEPRECATED API

**The Agents API is deprecated. Use the [Responses API](/docs/building_applications/responses) instead.**

The Responses API provides equivalent functionality with an OpenAI-compatible interface. New applications should use `client.responses.create()` rather than the agents workflow below.

:::

Types:

```python
from llama_stack_client.types import (
    InferenceStep,
    MemoryRetrievalStep,
    ShieldCallStep,
    ToolExecutionStep,
    ToolResponse,
    AgentCreateResponse,
)
```

Methods:

- <code title="post /v1/agents">client.agents.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/agents/agents.py">create</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/agent_create_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/agent_create_response.py">AgentCreateResponse</a></code>
- <code title="delete /v1/agents/{agent_id}">client.agents.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/agents/agents.py">delete</a>(agent_id) -> None</code>

### Session

Types:

```python
from llama_stack_client.types.agents import Session, SessionCreateResponse
```

Methods:

- <code title="post /v1/agents/{agent_id}/session">client.agents.session.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/agents/session.py">create</a>(agent_id, \*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/agents/session_create_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/agents/session_create_response.py">SessionCreateResponse</a></code>
- <code title="get /v1/agents/{agent_id}/session/{session_id}">client.agents.session.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/agents/session.py">retrieve</a>(session_id, \*, agent_id, \*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/agents/session_retrieve_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/agents/session.py">Session</a></code>
- <code title="delete /v1/agents/{agent_id}/session/{session_id}">client.agents.session.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/agents/session.py">delete</a>(session_id, \*, agent_id) -> None</code>

### Steps

Types:

```python
from llama_stack_client.types.agents import StepRetrieveResponse
```

Methods:

- <code title="get /v1/agents/{agent_id}/session/{session_id}/turn/{turn_id}/step/{step_id}">client.agents.steps.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/agents/steps.py">retrieve</a>(step_id, \*, agent_id, session_id, turn_id) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/agents/step_retrieve_response.py">StepRetrieveResponse</a></code>

### Turn

Types:

```python
from llama_stack_client.types.agents import Turn, TurnCreateResponse
```

Methods:

- <code title="post /v1/agents/{agent_id}/session/{session_id}/turn">client.agents.turn.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/agents/turn.py">create</a>(session_id, \*, agent_id, \*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/agents/turn_create_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/agents/turn_create_response.py">TurnCreateResponse</a></code>
- <code title="get /v1/agents/{agent_id}/session/{session_id}/turn/{turn_id}">client.agents.turn.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/agents/turn.py">retrieve</a>(turn_id, \*, agent_id, session_id) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/agents/turn.py">Turn</a></code>

## Datasets

Types:

```python
from llama_stack_client.types import (
    ListDatasetsResponse,
    DatasetRetrieveResponse,
    DatasetListResponse,
)
```

Methods:

- <code title="get /v1/datasets/{dataset_id}">client.datasets.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/datasets.py">retrieve</a>(dataset_id) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/dataset_retrieve_response.py">Optional[DatasetRetrieveResponse]</a></code>
- <code title="get /v1/datasets">client.datasets.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/datasets.py">list</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/dataset_list_response.py">DatasetListResponse</a></code>
- <code title="post /v1/datasets">client.datasets.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/datasets.py">register</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/dataset_register_params.py">params</a>) -> None</code>
- <code title="delete /v1/datasets/{dataset_id}">client.datasets.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/datasets.py">unregister</a>(dataset_id) -> None</code>

## Eval

Types:

```python
from llama_stack_client.types import EvaluateResponse, Job
```

Methods:

- <code title="post /v1/eval/tasks/{benchmark_id}/evaluations">client.eval.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/eval/eval.py">evaluate_rows</a>(benchmark_id, \*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/eval_evaluate_rows_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/evaluate_response.py">EvaluateResponse</a></code>
- <code title="post /v1/eval/tasks/{benchmark_id}/jobs">client.eval.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/eval/eval.py">run_eval</a>(benchmark_id, \*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/eval_run_eval_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/job.py">Job</a></code>

### Jobs

Types:

```python
from llama_stack_client.types.eval import JobStatusResponse
```

Methods:

- <code title="get /v1/eval/tasks/{benchmark_id}/jobs/{job_id}/result">client.eval.jobs.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/eval/jobs.py">retrieve</a>(job_id, \*, benchmark_id) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/evaluate_response.py">EvaluateResponse</a></code>
- <code title="delete /v1/eval/tasks/{benchmark_id}/jobs/{job_id}">client.eval.jobs.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/eval/jobs.py">cancel</a>(job_id, \*, benchmark_id) -> None</code>
- <code title="get /v1/eval/tasks/{benchmark_id}/jobs/{job_id}">client.eval.jobs.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/eval/jobs.py">status</a>(job_id, \*, benchmark_id) -> Optional[JobStatusResponse]</code>

## Inspect

Types:

```python
from llama_stack_client.types import HealthInfo, ProviderInfo, RouteInfo, VersionInfo
```

Methods:

- <code title="get /v1/health">client.inspect.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/inspect.py">health</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/health_info.py">HealthInfo</a></code>
- <code title="get /v1/version">client.inspect.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/inspect.py">version</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/version_info.py">VersionInfo</a></code>

## Inference

Types:

```python
from llama_stack_client.types import (
    CompletionResponse,
    EmbeddingsResponse,
    TokenLogProbs,
    InferenceChatCompletionResponse,
    InferenceCompletionResponse,
)
```

Methods:

- <code title="post /v1/inference/embeddings">client.inference.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/inference.py">embeddings</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/inference_embeddings_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/embeddings_response.py">EmbeddingsResponse</a></code>

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
from llama_stack_client.types import QueryChunksResponse
```

Methods:

- <code title="post /v1/vector-io/insert">client.vector_io.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/vector_io.py">insert</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/vector_io_insert_params.py">params</a>) -> None</code>
- <code title="post /v1/vector-io/query">client.vector_io.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/vector_io.py">query</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/vector_io_query_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/query_chunks_response.py">QueryChunksResponse</a></code>

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
from llama_stack_client.types import (
    ListVectorDBsResponse,
    VectorDBRetrieveResponse,
    VectorDBListResponse,
    VectorDBRegisterResponse,
)
```

Methods:

- <code title="get /v1/vector-dbs/{vector_db_id}">client.vector_dbs.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/vector_dbs.py">retrieve</a>(vector_db_id) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/vector_db_retrieve_response.py">Optional[VectorDBRetrieveResponse]</a></code>
- <code title="get /v1/vector-dbs">client.vector_dbs.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/vector_dbs.py">list</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/vector_db_list_response.py">VectorDBListResponse</a></code>
- <code title="post /v1/vector-dbs">client.vector_dbs.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/vector_dbs.py">register</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/vector_db_register_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/vector_db_register_response.py">VectorDBRegisterResponse</a></code>
- <code title="delete /v1/vector-dbs/{vector_db_id}">client.vector_dbs.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/vector_dbs.py">unregister</a>(vector_db_id) -> None</code>

## Models

Types:

```python
from llama_stack_client.types import ListModelsResponse, Model, ModelListResponse
```

Methods:

- <code title="get /v1/models/{model_id}">client.models.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/models.py">retrieve</a>(model_id) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/model.py">Optional[Model]</a></code>
- <code title="get /v1/models">client.models.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/models.py">list</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/model_list_response.py">ModelListResponse</a></code>
- <code title="post /v1/models">client.models.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/models.py">register</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/model_register_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/model.py">Model</a></code>
- <code title="delete /v1/models/{model_id}">client.models.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/models.py">unregister</a>(model_id) -> None</code>

## PostTraining

:::warning UNAVAILABLE API

**The Post Training API is not currently available in OGX.** There are no active providers implementing this API. The SDK types remain for forward compatibility but these endpoints are non-functional.

:::

Types:

```python
from llama_stack_client.types import ListPostTrainingJobsResponse, PostTrainingJob
```

Methods:

- <code title="post /v1/post-training/preference-optimize">client.post_training.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/post_training/post_training.py">preference_optimize</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/post_training_preference_optimize_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/post_training_job.py">PostTrainingJob</a></code>
- <code title="post /v1/post-training/supervised-fine-tune">client.post_training.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/post_training/post_training.py">supervised_fine_tune</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/post_training_supervised_fine_tune_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/post_training_job.py">PostTrainingJob</a></code>

### Job

Types:

```python
from llama_stack_client.types.post_training import (
    JobListResponse,
    JobArtifactsResponse,
    JobStatusResponse,
)
```

Methods:

- <code title="get /v1/post-training/jobs">client.post_training.job.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/post_training/job.py">list</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/post_training/job_list_response.py">JobListResponse</a></code>
- <code title="get /v1/post-training/job/artifacts">client.post_training.job.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/post_training/job.py">artifacts</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/post_training/job_artifacts_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/post_training/job_artifacts_response.py">Optional[JobArtifactsResponse]</a></code>
- <code title="post /v1/post-training/job/cancel">client.post_training.job.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/post_training/job.py">cancel</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/post_training/job_cancel_params.py">params</a>) -> None</code>
- <code title="get /v1/post-training/job/status">client.post_training.job.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/post_training/job.py">status</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/post_training/job_status_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/post_training/job_status_response.py">Optional[JobStatusResponse]</a></code>

## Providers

Types:

```python
from llama_stack_client.types import ListProvidersResponse, ProviderListResponse
```

Methods:

- <code title="get /v1/inspect/providers">client.providers.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/providers.py">list</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/provider_list_response.py">ProviderListResponse</a></code>

## Routes

Types:

```python
from llama_stack_client.types import ListRoutesResponse, RouteListResponse
```

Methods:

- <code title="get /v1/inspect/routes">client.routes.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/routes.py">list</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/route_list_response.py">RouteListResponse</a></code>

## Safety

Types:

```python
from llama_stack_client.types import RunShieldResponse
```

Methods:

- <code title="post /v1/safety/run-shield">client.safety.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/safety.py">run_shield</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/safety_run_shield_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/run_shield_response.py">RunShieldResponse</a></code>

## Shields

Types:

```python
from llama_stack_client.types import ListShieldsResponse, Shield, ShieldListResponse
```

Methods:

- <code title="get /v1/shields/{identifier}">client.shields.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/shields.py">retrieve</a>(identifier) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/shield.py">Optional[Shield]</a></code>
- <code title="get /v1/shields">client.shields.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/shields.py">list</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/shield_list_response.py">ShieldListResponse</a></code>
- <code title="post /v1/shields">client.shields.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/shields.py">register</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/shield_register_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/shield.py">Shield</a></code>

## SyntheticDataGeneration

:::warning UNAVAILABLE API

**The Synthetic Data Generation API is not currently available in OGX.** There are no active providers implementing this API. The SDK types remain for forward compatibility but these endpoints are non-functional.

:::

Types:

```python
from llama_stack_client.types import SyntheticDataGenerationResponse
```

Methods:

- <code title="post /v1/synthetic-data-generation/generate">client.synthetic_data_generation.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/synthetic_data_generation.py">generate</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/synthetic_data_generation_generate_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/synthetic_data_generation_response.py">SyntheticDataGenerationResponse</a></code>

## Datasetio

Types:

```python
from llama_stack_client.types import PaginatedRowsResult
```

Methods:

- <code title="post /v1/datasetio/rows">client.datasetio.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/datasetio.py">append_rows</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/datasetio_append_rows_params.py">params</a>) -> None</code>
- <code title="get /v1/datasetio/rows">client.datasetio.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/datasetio.py">get_rows_paginated</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/datasetio_get_rows_paginated_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/paginated_rows_result.py">PaginatedRowsResult</a></code>

## Scoring

Types:

```python
from llama_stack_client.types import ScoringScoreResponse, ScoringScoreBatchResponse
```

Methods:

- <code title="post /v1/scoring/score">client.scoring.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/scoring.py">score</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/scoring_score_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/scoring_score_response.py">ScoringScoreResponse</a></code>
- <code title="post /v1/scoring/score-batch">client.scoring.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/scoring.py">score_batch</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/scoring_score_batch_params.py">params</a>) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/scoring_score_batch_response.py">ScoringScoreBatchResponse</a></code>

## ScoringFunctions

Types:

```python
from llama_stack_client.types import (
    ListScoringFunctionsResponse,
    ScoringFn,
    ScoringFunctionListResponse,
)
```

Methods:

- <code title="get /v1/scoring-functions/{scoring_fn_id}">client.scoring_functions.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/scoring_functions.py">retrieve</a>(scoring_fn_id) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/scoring_fn.py">Optional[ScoringFn]</a></code>
- <code title="get /v1/scoring-functions">client.scoring_functions.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/scoring_functions.py">list</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/scoring_function_list_response.py">ScoringFunctionListResponse</a></code>
- <code title="post /v1/scoring-functions">client.scoring_functions.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/scoring_functions.py">register</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/scoring_function_register_params.py">params</a>) -> None</code>

## Benchmarks

Types:

```python
from llama_stack_client.types import (
    Benchmark,
    ListBenchmarksResponse,
    BenchmarkListResponse,
)
```

Methods:

- <code title="get /v1/eval-tasks/{benchmark_id}">client.benchmarks.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/benchmarks.py">retrieve</a>(benchmark_id) -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/benchmark.py">Optional[Benchmark]</a></code>
- <code title="get /v1/eval-tasks">client.benchmarks.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/benchmarks.py">list</a>() -> <a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/benchmark_list_response.py">BenchmarkListResponse</a></code>
- <code title="post /v1/eval-tasks">client.benchmarks.<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/resources/benchmarks.py">register</a>(\*\*<a href="https://github.com/meta-llama/llama-stack-client-python/tree/main/src/ogx_client/types/benchmark_register_params.py">params</a>) -> None</code>
