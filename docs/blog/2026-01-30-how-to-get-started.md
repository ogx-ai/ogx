---
slug: how-to-get-started-with-ogx
title: How to Get Started with OGX
authors: [cdoern, nathan-weinberg, ogx-team]
tags: [introduction, how-to]
date: 2026-01-30
---

There is no shortage of GenAI hosted services like OpenAI, Gemini, and Bedrock.

<!--truncate-->

Often, these services require tailoring your GenAI application directly to them, requiring developers to consider things that have nothing to do with their applications. OGX is an open source project aiming to standardize and offer a set of APIs for AI applications that stay the same, regardless of the backend services being provided via those APIs.

OGX’s APIs allow for a variety of use cases from running inference with Ollama on your laptop to a self-managed GPU system running inference with vLLM to a pure SaaS-based solution like Vertex. The standardized set of APIs each have providers that follow the same REST API implementation. An admin of the stack can specify which provider they want for each API and expose the REST API to users who get the same frontend experience regardless of the provider. This can allow you to run a single API surface layer using whatever Inference, Vector IO, or other solutions you may want while keeping your GenAI applications simple.

A OGX is defined by its `config.yaml` file which holds key information like which APIs you want to expose, which providers you want to initialize for those APIs, their configuration, and more. OGX features a CLI that allows you to launch and manage servers, either run locally on your machine or in a container!

Here is a sample portion of a `config.yaml`:

```yaml
version: 2
distro_name: starter
apis:
- agents
- batches
- datasetio
- eval
- files
- inference
- post_training
- safety
- scoring
- tool_runtime
- vector_io
providers:
inference:
- provider_id: ${env.OLLAMA_URL:+ollama}
provider_type: remote::ollama
config:
base_url: ${env.OLLAMA_URL:=http://localhost:11434/v1}
...

```

The current set of OGX APIs can be found here: [https://ogx-ai.github.io/docs/api-overview](https://ogx-ai.github.io/docs/api-overview)

All of these APIs, if set in the `config.yaml` which defines the stack to be stood up, will be available via a REST API with their initialized providers. Each API can have one or more providers and the `provider_id` can be specified at request time.

To get started quickly, all you need is OGX, Ollama, and your favorite inference model! For this example, we are using `gpt-oss:20b`.

If you already have Ollama installed as a service, you can simply pull the model:

```bash
ollama pull gpt-oss:20b
uv run --with ogx ogx list-deps --providers inference=remote::ollama --format uv | sh
uv run --with ogx ogx stack run --providers inference=remote::ollama

```

If you don't have Ollama running as a service, you can start it manually:

```bash
ollama serve > /dev/null 2>&1 &
ollama run gpt-oss:20b --keepalive 60m # you can exit this once the model is running due to --keepalive
uv run --with ogx ogx --providers inference=remote::ollama --format uv | sh
uv run --with ogx ogx stack run --providers inference=remote::ollama

```

Now you have Ollama running with `gpt-oss:20b`, and OGX running pointing to Ollama as the inference provider. This minimal setup is sufficient to connect to local Ollama and respond to `/v1/chat/completions` requests.

For a more feature-rich setup, you can use the starter distribution which gives you a full stack with additional APIs and providers:

```bash
ollama serve > /dev/null 2>&1 &
ollama run gpt-oss:20b --keepalive 60m # you can exit this once the model is running due to --keepalive
uv run --with ogx ogx list-deps starter --format uv | sh
export OLLAMA_URL=http://localhost:11434/v1
uv run --with ogx ogx stack run starter

```

A sample chat completion request would look like this:

```bash
curl -X POST http://localhost:8321/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "ollama/gpt-oss:20b",
"messages": [{"role": "user", "content": "Hello!"}]
}'

```

Notice, the model name must be prefixed with the `provider_id` in order for the request to route properly! In this example, we are just utilizing the `/chat/completions` route in the Inference API. The starter distribution has a large amount of APIs and ready to use providers baked in. Example API requests, similar to the one above, for other APIs can be found in the [OGX API specification](https://ogx-ai.github.io/docs/api/ogx-specification). Take it for a spin and see what you can do with OGX!
