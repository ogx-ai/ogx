---
slug: codex-cli-integration
title: "Use Codex CLI with Any Model Through OGX"
authors: [leseb, franciscojavierarceo]
tags: [codex, openai, cli, integration, tutorial]
date: 2026-05-19
---

OpenAI's [Codex CLI](https://github.com/openai/codex) is a terminal-native coding agent. It reads your codebase, proposes changes, runs commands, and iterates, all from your shell. The problem: it only talks to OpenAI's API.

OGX fixes that. By placing OGX between Codex and your inference provider, you get Codex's coding workflows with any model OGX supports: Llama via Ollama, Claude via Bedrock, Mistral via vLLM, or OpenAI itself with conversation compaction on top.

This post walks through setup, configuration, and what to expect from this alpha integration.

<!--truncate-->

## Why run Codex through OGX

Codex CLI is opinionated about its API surface. It speaks the OpenAI Responses API and expects a specific wire format. That's fine if you're paying OpenAI directly, but limiting if you want to:

- **Use open models** running on your own hardware via Ollama or vLLM
- **Route through a corporate proxy** that standardizes API access
- **Get conversation compaction** so long coding sessions don't blow past context limits
- **Switch models without reconfiguring Codex** by changing the OGX provider instead

OGX acts as a proxy that Codex already knows how to talk to. No patches to Codex, no forks. Just a config change.

## How the proxy chain works

The architecture is straightforward:

![Codex CLI to OGX to LLM Provider flow](./images/codex-cli-flow.svg)

Codex sends Responses API requests to OGX. OGX routes them to whatever inference provider you've configured, translating formats where needed. Codex doesn't know or care what's behind OGX.

This means tool execution (shell commands, file reads/writes, code generation) still happens locally in Codex. OGX only handles the inference routing.

## Setup

Three steps. Assumes you have OGX and Codex CLI already installed.

### 1. Start OGX

```bash
export OPENAI_API_KEY="your-key-here"
ogx stack run starter
```

For Ollama instead of OpenAI:

```bash
OLLAMA_URL=http://localhost:11434/v1 ogx stack run starter
```

### 2. Configure Codex CLI

Add the following to `~/.codex/config.toml`:

```toml
model = "openai/gpt-4o"
model_provider = "ogx"

[model_providers.ogx]
name = "OpenAI"
base_url = "http://localhost:8321/v1"
wire_api = "responses"
supports_websockets = false
```

If you're using Ollama, change the model line:

```toml
model = "ollama/llama3.2:3b"
```

### 3. Test it

```bash
codex "Write a hello world function in Python"
```

If you see Codex generate code and propose changes, the proxy is working.

## Model compatibility

Choose models that are exposed by your OGX server and compatible with the Responses API:

- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- `openai/gpt-5.4`
- `anthropic/claude-3-5-sonnet-20241022`
- `ollama/llama3.2:3b`

The model ID format depends on which inference provider is backing your OGX server. The prefix (e.g., `openai/`, `ollama/`) tells OGX where to route the request.

Since this setup uses `wire_api = "responses"`, ensure the selected model works with OGX's Responses path.

## What works, what doesn't

This integration is alpha. Here's the honest status.

**Works well:**

- Basic code generation and editing workflows
- Shell command execution and file operations
- Multi-turn conversations within a session
- Conversation compaction for long sessions
- Streaming responses

**Not yet supported:**

- **Memory persistence**: conversation history doesn't survive between Codex sessions. Each `codex` invocation starts fresh.
- **Error surfacing**: some provider-specific errors get swallowed by the proxy layer. If Codex hangs or returns empty, check OGX server logs.
- **Latency**: the extra network hop adds overhead. For local models (Ollama), this is negligible. For remote providers, it's a few hundred milliseconds on top of inference time.

## Troubleshooting

**"Model not found" errors:** check that the model ID includes the provider prefix. `gpt-4o` won't work, `openai/gpt-4o` will.

**Request compression errors:** Codex uses zstd compression by default. Make sure your OGX build supports request decompression. Recent versions do.

**Tool execution failures:** these happen in Codex, not OGX. Check that Codex has proper permissions for the operations it's trying to perform. OGX only handles inference.

**Empty or truncated responses:** check OGX server logs (`ogx` outputs to stderr by default). Provider-specific rate limits or token limits are the usual cause.

## What's next

We're working on closing the gaps in this integration:

- **Memory API integration** for persistent conversation storage across Codex sessions
- **Better error propagation** so provider failures surface clearly in the Codex UI
- **Performance optimizations** to reduce proxy overhead

For more details on OGX's provider architecture, see the [Providers Overview](https://ogx-ai.github.io/docs/providers). For Codex CLI documentation, visit the [Codex GitHub repository](https://github.com/openai/codex).
