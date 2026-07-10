---
slug: ogx-codex-cli
title: "Connect Codex CLI to Local and Hosted Models with OGX"
authors: [skamenan7]
tags: [ogx, codex, responses-api, vllm, ollama, agents]
date: 2026-06-30
---

Codex CLI brings an agent into the terminal, but teams often want that agent to use more than one model source. OGX gives Codex one OpenAI-compatible connection to local models, hosted models, and secured deployments.

With `ogx connect codex`, you can launch Codex against the models exposed by a running OGX server without hand-editing your normal Codex configuration. OGX discovers the available models, writes a temporary Codex session home, forwards auth and provider data when needed, and starts Codex with a session that points at OGX.

This post walks through the command, the generated Codex profile, and a local-first path with Ollama and vLLM. The result is a small but useful workflow: Codex gets a familiar Responses API endpoint, and teams keep model access, auth, and provider choice in OGX instead of hard-coding those details into every developer tool.

<!--truncate-->

## Why this is useful

Launching Codex is easy. The harder part starts when you want the same tool to work across different model backends: a local Ollama model, a vLLM server, a hosted OpenAI model, or an internal deployment with its own auth rules. Each one has a different endpoint, model name, and credential story. With OGX, Codex points at one OpenAI-compatible endpoint while OGX smoothly takes care of the backend work: finding models, routing requests, forwarding auth, and adapting provider-specific details.

OGX gives those tools a common front door:

- Codex talks to an OpenAI-compatible `/v1/responses` API.
- OGX exposes the models already registered in the running server.
- The provider behind each model can change without rewriting the Codex workflow.
- Auth and provider data can be forwarded through OGX instead of stored in a long-lived local Codex config.
- The generated Codex session home is temporary, so your existing `~/.codex/config.toml` stays untouched.

That makes OGX a good fit for agentic development across laptops, shared dev servers, and production-like environments. It is not just a proxy. It is a control plane for model access, auth shape, provider routing, and day-to-day debugging.

Codex uses OGX through the Responses API, but the provider setup is useful beyond Codex. Once a model provider such as vLLM, Ollama, or OpenAI is registered in OGX, other OpenAI-compatible clients can use the same model access layer through chat completions or completions when those endpoints are enabled. That includes Python scripts using the OpenAI SDK, notebooks, OpenCode, LiteLLM or LangChain-style apps, and internal tools that already speak OpenAI-compatible APIs. Teams can configure routing and auth once in OGX, then reuse that setup from Codex and from simpler inference clients.

## What the command does

The command is intentionally small:

```bash
ogx connect codex
```

Behind that command, OGX does the mechanical work that is easy to get wrong by hand:

1. Queries `GET /v1/models` on the running OGX server.
2. Filters out embedding models.
3. Selects the requested model, or the first available LLM model.
4. Creates a temporary `CODEX_HOME`.
5. Writes `ogx.config.toml` and `ogx-model-catalog.json`.
6. Launches `codex -p ogx`, or `codex exec -p ogx` when `--exec` is used.

The generated profile uses the Responses wire API:

```toml
model_provider = "ogx"

[features]
multi_agent = false

[model_providers.ogx]
wire_api = "responses"
```

The `multi_agent` flag is disabled because current OGX Responses models do not accept the Codex `namespace` tool shape. That keeps the default connector path focused on the request shape OGX can handle today.

The flow looks like this:

```text
Codex CLI
  |
  | generated OGX profile
  v
OGX /v1/responses
  |
  | provider routing
  v
OpenAI, vLLM, Ollama, Bedrock, or another OGX-backed model provider
```

The important part is the boundary: Codex only needs to know about the OGX profile. OGX owns the provider mapping.

## Try it with a running OGX server

Start an OGX server that exposes at least one LLM model. The starter distribution enables providers from environment variables, so you can begin with a local provider instead of an OpenAI key.

Start Ollama in one terminal:

```bash
ollama serve
```

Then pull a model and start OGX in another terminal:

```bash
ollama pull llama3.2:3b
export OLLAMA_URL="http://localhost:11434/v1"
uv run ogx run starter
```

If you do want to use an OpenAI-backed model instead, set `OPENAI_API_KEY` before starting the same starter distribution.

Then launch Codex in another terminal:

```bash
uv run ogx connect codex
```

If you want a specific model from the OGX model list:

```bash
uv run ogx connect codex \
  --model ollama/llama3.2:3b
```

For a quick non-interactive run, use `--exec`:

```bash
uv run ogx connect codex \
  --model ollama/llama3.2:3b \
  --exec "Explain in one sentence why OGX is useful with Codex CLI."
```

This is useful when you want a quick answer before opening an interactive Codex session. Codex sends the prompt through OGX, OGX routes it to the selected model, and the final answer comes back through the same Responses API path.

## Auth and provider data

The same setup works when OGX is running behind an authenticated endpoint. Codex still talks to one OGX URL, while OGX enforces the same access policy, provider routing, and credential handling that other clients use.

That matters in shared environments. A team can let Codex use approved models without putting long-lived provider secrets in `~/.codex/config.toml`, and without teaching Codex every backend-specific auth shape. If the OGX server requires bearer auth, set `OGX_API_KEY`. If a provider path needs request-scoped data, such as a passthrough provider token, set `OGX_PROVIDER_DATA`.

```bash
export OGX_API_KEY="your-ogx-access-token"
export OGX_PROVIDER_DATA='{"passthrough_api_key":"provider-token"}'

uv run ogx connect codex \
  --url https://ogx.example.com/v1
```

When `OGX_API_KEY` is set, Codex uses it to authenticate to OGX. When `OGX_PROVIDER_DATA` is set, OGX receives that JSON on each request and can use it for provider-specific needs such as passthrough credentials, tenant context, or other request-scoped routing data.

Codex does not need to know what the backend provider expects. It only points at OGX for the current session; OGX decides how that request is authorized and how provider-specific data is applied before the request reaches the model.

## Try it with a vLLM-backed OGX server

vLLM is useful when you want Codex to use an open model served from your own environment. It exposes an OpenAI-compatible server that OGX can register as `remote::vllm`, so the same Codex workflow can run against a vLLM-backed model instead of a hosted provider.

Start a vLLM OpenAI-compatible server:

```bash
export VLLM_API_TOKEN="fake"

vllm serve Qwen/Qwen3-8B \
  --api-key "$VLLM_API_TOKEN"
```

Then start OGX with the stock starter distribution pointed at that vLLM server in another terminal:

```bash
export VLLM_URL="http://localhost:8000/v1"
export VLLM_API_TOKEN="fake"

uv run ogx run starter
```

The starter distribution enables the vLLM provider when `VLLM_URL` is set. The relevant provider configuration is:

```yaml
providers:
  inference:
  - provider_id: ${env.VLLM_URL:+vllm}
    provider_type: remote::vllm
    config:
      base_url: ${env.VLLM_URL:=}
      max_tokens: ${env.VLLM_MAX_TOKENS:=4096}
      api_token: ${env.VLLM_API_TOKEN:=fake}
      network:
        tls:
          verify: ${env.VLLM_TLS_VERIFY:=true}
```

Now connect Codex through OGX:

```bash
uv run ogx connect codex \
  --model vllm/Qwen/Qwen3-8B \
  --exec "Explain in one sentence how this Codex request is reaching the vLLM model."
```

That is the main point: Codex still uses the same OGX command, while OGX changes the provider behind the model ID. Ollama is a good first local path for basic text flows. vLLM is useful when you want an OpenAI-compatible server for open models. Provider and model compatibility still matter, but those details stay behind the OGX boundary instead of becoming permanent Codex configuration.

## Summary

`ogx connect codex` makes Codex feel like a first-class OGX client:

- It discovers models from the running OGX server.
- It generates a temporary Codex profile and model catalog.
- It keeps your normal Codex config untouched.
- It supports bearer auth and provider-data forwarding.
- It works for interactive Codex sessions and quick non-interactive runs.
- It works with local providers such as Ollama and vLLM as well as hosted providers.
- It keeps current alpha limits explicit, including no persistent Codex memory and disabled Codex multi-agent tools.

The practical takeaway: OGX gives teams one place to manage model access while still letting developer tools like Codex move fast. If you can start OGX and see your model in `/v1/models`, you have a clear path to using that model from Codex and debugging each layer with concrete evidence.
