# agents / responses (inline provider)

The `builtin` agents provider implements the OpenAI Responses API. This directory is being renamed from `agents` to `responses` (see PR #5195).

## Directory Structure

```text
agents/
  builtin/
    __init__.py        # Provider factory (get_provider_impl)
    agents.py          # Core orchestration logic
    config.py          # BuiltinAgentsImplConfig
    safety.py          # Safety checking integration
    responses/         # OpenAI Responses API implementation
      __init__.py
      openai_responses.py   # Responses API handler
      streaming.py          # SSE streaming for responses
      tool_executor.py      # Tool call execution engine
      types.py              # Response-specific types
      utils.py              # Response utilities
  __init__.py
```

## What It Does

This provider handles:

- **Agent turns**: Multi-step inference with tool calling loops. The agent calls the inference provider, executes any requested tools, feeds results back, and repeats until the model produces a final response.
- **OpenAI Responses API**: Implements the `/v1/responses` endpoint, which provides a stateful, agentic interface compatible with OpenAI's Responses API. Supports built-in tools (web search, code interpreter, file search) and custom function tools.
- **Safety integration**: Optionally runs input and output through safety shields before and after inference.

## Dependencies

This provider depends on:

- `Api.inference` -- for LLM calls
- `Api.safety` -- for input/output safety checks
- `Api.tool_runtime` -- for executing tool calls
- `Api.vector_io` -- for file search / RAG
- `Api.files` -- for file management
