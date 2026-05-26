---
slug: claude-code-integration
title: "Using Claude Code with Any Model via OGX"
authors: [leseb, cdoern]
tags: [claude-code, anthropic, integration, tutorial, vllm, ollama, openai]
date: 2026-06-02
---

Claude Code is one of the best coding assistants available. But what if you want to use it with GPT-4o, Qwen, Llama, or a model running on your own hardware? OGX makes that possible. Point Claude Code at an OGX server and use any model from any provider — local or cloud — without changing how you use Claude Code.

This post walks through the setup, explains how the translation works under the hood, and shows how to configure multi-provider routing so different Claude Code model tiers hit different backends.

<!--truncate-->

## The idea

Claude Code talks to the Anthropic Messages API (`/v1/messages`). OGX implements that API. When Claude Code sends a request, OGX receives it, translates the format if needed, and forwards it to whatever inference provider you've configured — OpenAI, vLLM, Ollama, Fireworks, Groq, Bedrock, or any of the other [supported providers](https://ogx-ai.github.io/docs/providers).

![Claude Code integration flow](/img/claude-code-flow.svg)

The translation layer handles message format conversion, tool call transformations, and streaming event reformatting. For providers that already support the Messages API natively (Ollama and vLLM with compatible models), OGX passes requests through directly — no translation overhead.

## Quick start

Three steps. Five minutes.

### 1. Start OGX

Pick your provider and start the server:

```bash
# With OpenAI
export OPENAI_API_KEY="your-key-here"
ogx stack run starter

# With vLLM
export VLLM_URL="http://localhost:8000/v1"
ogx stack run starter

# With Ollama
export OLLAMA_URL="http://localhost:11434/v1"
ogx stack run starter
```

### 2. Configure Claude Code

Point Claude Code at your OGX server:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8321"
export ANTHROPIC_API_KEY="fake"  # Not validated when using local providers
```

> **Important:** If Claude Code is configured to use Vertex AI or Bedrock (e.g., `CLAUDE_CODE_USE_VERTEX=1` is set), it will ignore `ANTHROPIC_BASE_URL`. You need to unset those variables first:
>
> ```bash
> unset CLAUDE_CODE_USE_VERTEX
> unset ANTHROPIC_VERTEX_PROJECT_ID
> ```

### 3. Use it

```bash
claude "Write a hello world function in Python"
claude "Create a Flask app with user authentication"
```

That's it. Claude Code doesn't know it's not talking to Anthropic's servers.

## How model routing works

Claude Code sends requests using Claude model names internally (e.g., `claude-haiku-4-5-20251001`). The OGX starter distribution automatically registers aliases that map these names to whatever backend model you've configured:

```yaml
# Pre-configured in starter config.yaml
registered_resources:
  models:
  - model_id: claude-haiku-4-5-20251001
    provider_id: "all"
    provider_model_id: "auto"
    model_type: llm
```

You control which backend model handles each tier via environment variables — Claude Code doesn't need to know anything about the underlying model.

## Multi-provider routing

This is where it gets interesting. You can map different Claude Code model tiers to different backends — fast local models for quick tasks, cloud APIs for complex reasoning:

```bash
# Fast model → local vLLM
export ANTHROPIC_DEFAULT_HAIKU_MODEL="vllm/Qwen/Qwen3-8B"

# Balanced model → OpenAI GPT-4o
export ANTHROPIC_DEFAULT_SONNET_MODEL="openai/gpt-4o"

# Most capable → OpenAI o1
export ANTHROPIC_DEFAULT_OPUS_MODEL="openai/o1"
```

Claude Code routes automatically based on which model tier it uses internally:

```bash
claude "Quick task"    # Uses haiku → vLLM (local)
claude "Complex task"  # Uses sonnet → OpenAI GPT-4o
```

This gives you fine-grained control over cost and latency without changing your workflow.

## Provider setup examples

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_BASE_URL="http://localhost:8321"
export ANTHROPIC_API_KEY="fake"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="openai/gpt-4o-mini"
export ANTHROPIC_DEFAULT_SONNET_MODEL="openai/gpt-4o"

ogx stack run starter
claude "Implement a binary search tree"
```

### vLLM with Qwen

```bash
# Start vLLM server
vllm serve Qwen/Qwen3-8B --api-key fake

# Start OGX
export VLLM_URL="http://localhost:8000/v1"
export ANTHROPIC_BASE_URL="http://localhost:8321"
export ANTHROPIC_API_KEY="fake"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="vllm/Qwen/Qwen3-8B"

ogx stack run starter
claude "Write a Fibonacci function"
```

### Ollama with Llama

```bash
ollama serve
ollama pull llama3.3:70b

export OLLAMA_URL="http://localhost:11434/v1"
export ANTHROPIC_BASE_URL="http://localhost:8321"
export ANTHROPIC_API_KEY="fake"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="ollama/llama3.3:70b"

ogx stack run starter
claude "Explain quicksort"
```

## What's supported

All core Claude Code features work through OGX:

- **Multi-turn conversations** with system messages and streaming
- **Tool use** — file operations, shell commands, code execution (these run in Claude Code's runtime, not OGX)
- **Extended thinking** — thinking blocks for reasoning transparency
- **Token counting** via `/v1/messages/count_tokens`
- **Prompt caching** when using providers that support it

Provider capabilities differ:

| Provider | Native Messages API | Thinking Support | Prompt Caching |
|----------|-------------------|------------------|----------------|
| OpenAI | ❌ (translated) | ⚠️ (via reasoning) | ❌ |
| vLLM | ✅ | ❌ | ❌ |
| Ollama | ✅ | ❌ | ❌ |
| Bedrock, Fireworks, Groq, Together | ❌ (translated) | ❌ | ❌ |

## Advanced: custom model mappings

For more control over how Claude model names map to providers, register models explicitly via the API:

```bash
curl http://localhost:8321/v1/models \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "claude-haiku-4-5-20251001",
    "provider_id": "vllm",
    "provider_model_id": "Qwen/Qwen3-8B",
    "model_type": "llm"
  }'
```

Or declaratively in `config.yaml`:

```yaml
registered_resources:
  models:
  - model_id: claude-haiku-4-5-20251001
    provider_id: vllm
    provider_model_id: Qwen/Qwen3-8B
    model_type: llm
```

## Claude Agent SDK

If you're building custom agents with the Claude Agent SDK, OGX works as a drop-in backend:

```python
from claude_agent_sdk import Agent

agent = Agent(
    base_url="http://localhost:8321",
    api_key="fake",
    model="vllm/Qwen/Qwen3-8B",
)

response = agent.send("Write a function to parse CSV files")
```

## Troubleshooting

**Claude Code ignores `ANTHROPIC_BASE_URL`** — If you see errors about your "vertex deployment" or "bedrock", Claude Code is using a cloud provider and bypassing the base URL entirely. Unset the relevant variables:

```bash
unset CLAUDE_CODE_USE_VERTEX
unset ANTHROPIC_VERTEX_PROJECT_ID
# For Bedrock users:
unset CLAUDE_CODE_USE_BEDROCK
```

**"Model not found" errors** — Set the model mapping environment variable so OGX knows which backend model to use for each Claude model tier:

```bash
export ANTHROPIC_DEFAULT_HAIKU_MODEL="your-provider/your-model"
```

**Authentication errors with local providers** — Set a dummy API key:

```bash
export ANTHROPIC_API_KEY="fake"
```

**Slow cloud provider responses** — Expected. Claude Code → OGX → provider adds a network hop. Use local providers (vLLM, Ollama) for lower latency. The format translation itself adds only ~5-20ms.

**Tool use not working** — Tool execution happens in Claude Code's runtime, not OGX. Make sure Claude Code has the right permissions and your model supports tool use.

## What's next

The [full documentation](https://ogx-ai.github.io/docs/building_applications/claude_code_integration) covers additional details on performance tuning and provider-specific configuration. If you run into issues or want to improve the integration, [open an issue](https://github.com/ogx-ai/ogx/issues) or join us on [Discord](https://discord.gg/ZAFjsrcw).
