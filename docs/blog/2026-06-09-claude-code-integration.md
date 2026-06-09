---
slug: claude-code-integration
title: "Using Claude Code with Any Model via OGX"
authors: [leseb, cdoern]
tags: [claude-code, anthropic, integration, tutorial, vllm, ollama, openai]
date: 2026-06-09
---

Claude Code is one of the best coding assistants available. But what if you want to use it with GPT-4o, Qwen, Llama, or a model running on your own hardware? OGX makes that possible. A single command connects Claude Code to your OGX server, auto-discovers your models, and maps them to Claude's haiku/sonnet/opus tiers.

This post walks through the setup, explains how the translation works under the hood, and shows how to configure multi-provider routing so different Claude Code model tiers hit different backends.

<!--truncate-->

## The idea

Claude Code talks to the Anthropic Messages API (`/v1/messages`). OGX implements that API. When Claude Code sends a request, OGX receives it, translates the format if needed, and forwards it to whatever inference provider you've configured — OpenAI, vLLM, Ollama, Fireworks, Groq, Bedrock, or any of the other [supported providers](https://ogx-ai.github.io/docs/providers).

![Claude Code integration flow](/img/claude-code-flow.svg)

The translation layer handles message format conversion, tool call transformations, streaming event reformatting, and extended thinking (including signature deltas and redacted thinking blocks). For providers that already support the Messages API natively (Ollama and vLLM with compatible models), OGX passes requests through directly — no translation overhead.

## Quick start

Two commands. Two minutes.

### 1. Start OGX

Pick your provider and start the server:

```bash
# With OpenAI
export OPENAI_API_KEY="your-key-here"
ogx run starter

# With vLLM
export VLLM_URL="http://localhost:8000/v1"
ogx run starter

# With Ollama
export OLLAMA_URL="http://localhost:11434/v1"
ogx run starter
```

### 2. Connect Claude Code

```bash
ogx connect claude
```

That's it. The command queries your OGX server for available models, maps them to Claude's haiku/sonnet/opus tiers, sets the right environment variables (including unsetting any Vertex/Bedrock variables that would bypass OGX), and launches Claude Code.

### What `ogx connect claude` does

```text
ogx connect claude
       |
       v
  GET /v1/models (discover available models)
       |
       v
  Map models to Claude tiers (haiku/sonnet/opus)
       |
       v
  Launch claude with ANTHROPIC_BASE_URL + tier env vars
```

No manual environment variable setup. No remembering which model names map to which tiers. No Vertex/Bedrock conflicts.

## Model configuration

### Default behavior

With no flags, `ogx connect claude` maps all three Claude tiers to the first available LLM model on your OGX server.

### One model for all tiers

```bash
ogx connect claude --model openai/gpt-4o
```

### Different models per tier

This is the real power — route fast tasks to cheap local models and complex reasoning to cloud APIs:

```bash
ogx connect claude \
  --haiku-model openai/gpt-4o-mini \
  --sonnet-model openai/gpt-4o \
  --opus-model openai/o1
```

### Shell integration with `--print-env`

Instead of launching Claude Code, print the environment variables for manual use:

```bash
eval "$(ogx connect claude --print-env --model openai/gpt-4o)"
claude "Hello world"
```

### Forwarding arguments to Claude Code

Anything after `--` is passed through to `claude`:

```bash
ogx connect claude -- -p "Write a hello world function"
```

## Provider setup examples

### OpenAI

```bash
# Terminal 1
export OPENAI_API_KEY="sk-..."
ogx run starter

# Terminal 2
ogx connect claude --model openai/gpt-4o
```

### vLLM with Qwen

```bash
# Start vLLM
vllm serve Qwen/Qwen3-8B --api-key fake

# Terminal 1
export VLLM_URL="http://localhost:8000/v1"
ogx run starter

# Terminal 2
ogx connect claude --model vllm/Qwen/Qwen3-8B
```

### Ollama with Llama

```bash
ollama serve
ollama pull llama3.3:70b

# Terminal 1
export OLLAMA_URL="http://localhost:11434/v1"
ogx run starter

# Terminal 2
ogx connect claude --model ollama/llama3.3:70b
```

### Multiple providers with per-tier routing

```bash
# Terminal 1
export VLLM_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="sk-..."
ogx run starter

# Terminal 2
ogx connect claude \
  --haiku-model vllm/Qwen/Qwen3-8B \
  --sonnet-model openai/gpt-4o \
  --opus-model openai/o1
```

## What's supported

All core Claude Code features work through OGX:

- **Multi-turn conversations** with system messages and streaming
- **Tool use** — file operations, shell commands, code execution (these run in Claude Code's runtime, not OGX)
- **Extended thinking** — full support including signature deltas and redacted thinking blocks in passthrough mode; clear error when attempting thinking in translation mode
- **Token counting** via `/v1/messages/count_tokens`
- **Prompt caching** — `cache_control` breakpoints from the Anthropic SDK are forwarded correctly in passthrough mode
- **Any inference provider** — OpenAI, vLLM, Ollama, Fireworks, Together, Groq, Bedrock, etc.

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

## Manual setup (without `ogx connect claude`)

If you prefer to configure environment variables yourself:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8321"
export ANTHROPIC_AUTH_TOKEN="ogx"

# Map Claude model tiers to your backend models
export ANTHROPIC_DEFAULT_HAIKU_MODEL="openai/gpt-4o-mini"
export ANTHROPIC_DEFAULT_SONNET_MODEL="openai/gpt-4o"
export ANTHROPIC_DEFAULT_OPUS_MODEL="openai/o1"

# Unset any Vertex/Bedrock variables
unset CLAUDE_CODE_USE_VERTEX
unset ANTHROPIC_VERTEX_PROJECT_ID
unset CLAUDE_CODE_USE_BEDROCK

claude "Write a hello world function in Python"
```

## Troubleshooting

**`max_tokens` errors with OpenAI models** — Claude Code requests token limits based on Claude model specs, which may exceed what the backend model supports. Use a model with higher token limits, or use per-tier flags to route different workloads appropriately.

**"Failed to connect to OGX server"** — The OGX server isn't running or isn't reachable. Start it with `ogx run starter`.

**"Failed to find any LLM models"** — The server is running but has no LLM models registered. Check your distribution config and ensure at least one inference provider is configured.

**Claude Code ignores `ANTHROPIC_BASE_URL` (manual setup only)** — If `CLAUDE_CODE_USE_VERTEX=1` or similar is set, Claude Code bypasses `ANTHROPIC_BASE_URL`. The `ogx connect claude` command handles this automatically. For manual setup, unset those variables first.

**Slow cloud provider responses** — Expected. Claude Code → OGX → provider adds a network hop. Use local providers (vLLM, Ollama) for lower latency. The format translation itself adds only ~5-20ms.

**Tool use not working** — Tool execution happens in Claude Code's runtime, not OGX. Make sure Claude Code has the right permissions and your model supports tool use.

## What's next

The [full documentation](https://ogx-ai.github.io/docs/building_applications/claude_code_integration) covers the complete CLI reference, performance tuning, and provider-specific configuration. If you run into issues or want to improve the integration, [open an issue](https://github.com/ogx-ai/ogx/issues) or join us on [Discord](https://discord.gg/ZAFjsrcw).
