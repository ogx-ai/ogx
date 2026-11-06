# tool_runtime (inline providers)

Inline tool runtime providers that execute tools within the server process.

## Directory Structure

```text
tool_runtime/
  file_search/            # file_search tool runtime
    __init__.py           # Provider factory (get_provider_impl)
    config.py              # FileSearchToolRuntimeConfig
    context_retriever.py   # Query rewriting for file_search
    file_search.py         # FileSearchToolRuntimeImpl (query() against vector stores)
  __init__.py
```

## file_search Tool Runtime

The `file_search` provider implements the `file_search` built-in tool. When an agent invokes this tool:

1. The query is sent to the configured vector store(s) (via `Api.vector_io`).
2. Relevant document chunks are retrieved based on embedding similarity.
3. Retrieved chunks are assembled into context and returned to the agent.

This provider depends on `Api.vector_io` for vector storage and `Api.inference` for generating embeddings.
Document ingestion happens separately, through the vector store file-attach path
(`Api.vector_io`'s `openai_attach_file_to_vector_store`), not through this provider.

## Tool Runtime vs. Tool Groups

- **Tool groups** (`Api.tool_groups`) are a routing table that maps tool group identifiers to tool runtime providers.
- **Tool runtime** (`Api.tool_runtime`) is the actual implementation that executes tools. The `ToolRuntimeRouter` in `core/routers/tool_runtime.py` uses the tool groups routing table to dispatch tool execution requests.
