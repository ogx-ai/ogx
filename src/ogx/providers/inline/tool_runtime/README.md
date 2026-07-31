# tool_runtime (inline providers)

Inline tool runtime providers that execute tools within the server process.

## Directory Structure

```text
tool_runtime/
  rag/                 # RAG (Retrieval-Augmented Generation) tool runtime
    __init__.py        # Provider factory (get_provider_impl)
    config.py          # RAGToolRuntimeConfig
    context_retriever.py  # Document retrieval and context assembly
    memory.py          # Vector store integration for document storage
  __init__.py
```

## RAG Tool Runtime

The `rag` provider implements the `knowledge_search` built-in tool. When an agent invokes this tool:

1. The query is sent to the configured vector store (via `Api.vector_io`).
2. Relevant document chunks are retrieved based on embedding similarity.
3. Retrieved chunks are assembled into context and returned to the agent.

This provider depends on `Api.vector_io` for vector storage and `Api.inference` for generating embeddings.

## Tool Runtime vs. Tool Groups

- **Tool groups** (`Api.tool_groups`) are a routing table that maps tool group identifiers to tool runtime providers.
- **Tool runtime** (`Api.tool_runtime`) is the actual implementation that executes tools. The `ToolRuntimeRouter` in `core/routers/tool_runtime.py` uses the tool groups routing table to dispatch tool execution requests.
