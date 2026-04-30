---
slug: opencode-blog
title: "OpenCode ❤️ OGX"
authors: [nathan-weinberg]
tags: []
date: 2026-05-05
---

[OpenCode](https://opencode.ai/) is an open source AI coding agent that helps you write code in your terminal, IDE, or desktop. It is a popular open source alternative
for tools like Claude Code and Codex.

OpenCode has a concept of [providers](https://opencode.ai/docs/providers/) that is similar to OGX's inference providers - they are a local or cloud-based model inference endpoint that expose an LLM for OpenCode to utilize. This is similar but does differ from [OGX providers](https://ogx-ai.github.io/docs/providers) which are inclusive of inference but also include providers for vector stores, safety backends, tool runtimes, etc.

OGX as a OpenCode provider has some strong advantages over providers that offer only inference:

- Unified API for tools + RAG + storage
- Multiple providers behind one endpoint
- Built-in orchestration layer

In this blog I am going to share how to start running OpenCode with OGX as a provider, using OpenCode's [custom provider](https://opencode.ai/docs/providers/#custom-provider) feature.

The blog assumes you already have an OGX server up and running - see our [Getting Started guide](https://ogx-ai.github.io/docs/getting_started/quickstart) to learn more.

## Download OpenCode

Downloading OpenCode is simple and can be done in various ways. You can see a full list of methods [here](https://opencode.ai/download) but generally the below `curl` command is suifficient in most cases.

```bash
curl -fsSL https://opencode.ai/install | bash
```

## Configure OGX as a provider for OpenCode

As mentioned before, this blog assumes an OGX server is already running at `localhost:8321` - in this case, we are also making the following assumptions:

- The `remote::vllm` provider is enabled, serving the `Qwen/Qwen3-8B` model
- The `remote::watsonx` provider is enabled, with the `gpt-oss-120b` model available
- No authentication has been added

You can verify what models your OGX server has available with `curl http://localhost:8321/v1/models`

We can now configure OpenCode to use our OGX server via a custom provider.

Create a file `~/.config/opencode/opencode.json` with the following content:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ogx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OGX",
      "options": {
        "baseURL": "http://localhost:8321/v1"
      },
      "models": {
        "vllm-inference/Qwen/Qwen3-8B": {
          "name": "Qwen3-8B"
        },
        "watsonx/openai/gpt-oss-120b": {
          "name": "gpt-oss-120b"
        }
      }
    }
  }
}
```

Once the file is created, start OpenCode - it should look something like this:

![OpenCode Home](./images/opencode-home.png)

Run `/connect` in the TUI. If you search `OGX` the provider should come up with our two models listed.

![OpenCode Models](./images/opencode-models.png)

Select `gpt-oss-120b` and hit enter. If you are prompted for an API key, you can just put `None` since we haven't configured one in this case.

## Use OpenCode with OGX

Now that you have your OGX-provided model selected, it's time to start using OpenCode with OGX! Go ahead and start prompting in the TUI - it should look something like this:

![OpenCode User](./images/opencode-user.png)

And that is it! You can tweak your OGX server and OpenCode custom provider configuration to add additional providers, models, and whatever else you might need for yourself or your enterprise.

Thanks for reading, and happy coding!
