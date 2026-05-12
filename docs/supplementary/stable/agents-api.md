## Responses API

The Responses API provides OpenAI-compatible functionality with enhanced capabilities for dynamic, stateful interactions.

> **✅ STABLE**: This API is production-ready with backward compatibility guarantees. Recommended for production applications.

### ✅ Supported Tools

The Responses API supports the following tool types:

- **`web_search`**: Search the web for current information and real-time data
- **`file_search`**: Search through uploaded files and vector stores
  - Supports dynamic `vector_store_ids` per call
  - Compatible with OpenAI file search patterns
- **`function`**: Call custom functions with JSON schema validation
- **`mcp_tool`**: Model Context Protocol integration

### ✅ Supported Fields & Features

**Core Capabilities:**

- **Dynamic Configuration**: Switch models, vector stores, and tools per request without pre-configuration
- **Conversation Branching**: Use `previous_response_id` to branch conversations and explore different paths
- **Rich Annotations**: Automatic file citations, URL citations, and container file citations
- **Status Tracking**: Monitor tool call execution status and handle failures gracefully

### 🚧 Work in Progress

- Full real-time response streaming support
- `tool_choice` parameter
- `max_tool_calls` parameter
- Built-in tools (code interpreter, containers API)
- Moderation & guardrails
- `reasoning` capabilities
- `service_tier`
- `logprobs`
- `max_output_tokens`
- `metadata` handling
- `instructions`
- `incomplete_details`
- `background`
