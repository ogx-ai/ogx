<!-- markdownlint-disable MD036 -->
# OGX 2.0 Release Notes

**Release Date:** November 2026

**OGX 2.0** is a major release from 28 contributors (133 PRs since v1.5.0). New in this release: the SystemOne judgment model API, available across Ollama, llama.cpp, SGLang, and vLLM; a new SGLang inference provider with Tev1 support; namespaced function tools; a new SearchApi web search backend; and Python 3.14 support.

The headline themes for 2.0:

- **SystemOne judgment model API** — a first-class judgment surface with the TypeSafe provider and local-engine judgment support on Ollama, llama.cpp, SGLang, and vLLM.
- **Web search as a routed API** — search moved out of the `tool_runtime` tools into the `web_search` endpoint, with a new SearchApi backend and a built-in placeholder so it works before any search provider is configured.
- **`tool_runtime`/`tool_groups` retirement** — web and file search now run inside the Responses API, and MCP tools are served by the Responses API directly.
- **Namespaced function tools** with identity-exact mapping through the chat-completions layer.
- **SGLang provider** with Tev1 support and integration CI enabled.
- **Python 3.13–3.14** (3.12 dropped).

## Overview

### Local engine support

- **New SGLang inference provider** with Tev1 support and integration CI enabled ([#6776](https://github.com/ogx-ai/ogx/pull/6776), [#6822](https://github.com/ogx-ai/ogx/pull/6822))
- **Python 3.13–3.14** (3.12 dropped) ([#6973](https://github.com/ogx-ai/ogx/pull/6973))
- Engine pins advanced across the board: **llama.cpp b12360 (v1.0.0)** ([#7010](https://github.com/ogx-ai/ogx/pull/7010)), **Ollama 0.40.16** ([#7011](https://github.com/ogx-ai/ogx/pull/7011)), **vLLM 0.32.0** ([#6840](https://github.com/ogx-ai/ogx/pull/6840)), **SGLang 0.5.23** ([#6986](https://github.com/ogx-ai/ogx/pull/6986))
- Ollama embedding models classified through models.dev ([#6879](https://github.com/ogx-ai/ogx/pull/6879))
- Self-hosted backend and reasoning suites unified into single test matrices ([#6881](https://github.com/ogx-ai/ogx/pull/6881), [#6887](https://github.com/ogx-ai/ogx/pull/6887), [#6889](https://github.com/ogx-ai/ogx/pull/6889))

### Harness & agent support

- **Web search is now a routed API** — search moved from the tool_runtime tools to the `web_search` endpoint; new SearchApi backend, configurable Brave Search base URL, and a built-in placeholder so it works before any search provider is configured ([#7014](https://github.com/ogx-ai/ogx/pull/7014), [#6927](https://github.com/ogx-ai/ogx/pull/6927), [#6878](https://github.com/ogx-ai/ogx/pull/6878), [#6957](https://github.com/ogx-ai/ogx/pull/6957))
- **`tool_runtime`/`tool_groups` APIs retired** — web and file search now run inside the Responses API, and MCP tools are served by the Responses API directly, optionally via MCP connectors ([#7039](https://github.com/ogx-ai/ogx/pull/7039), [#7033](https://github.com/ogx-ai/ogx/pull/7033))
- **Namespaced function tools** with identity-exact mapping through the chat-completions layer ([#6941](https://github.com/ogx-ai/ogx/pull/6941))
- **`ogx connect` CLI overhaul** — shared `--url`, `-k/--insecure`, API-key pass-through, `https://localhost` default, consistent argument forwarding across the agent connectors; deprecated shims removed ([#6928](https://github.com/ogx-ai/ogx/pull/6928), [#6930](https://github.com/ogx-ai/ogx/pull/6930), [#6942](https://github.com/ogx-ai/ogx/pull/6942), [#6953](https://github.com/ogx-ai/ogx/pull/6953), [#6954](https://github.com/ogx-ai/ogx/pull/6954), [#6929](https://github.com/ogx-ai/ogx/pull/6929), [#7076](https://github.com/ogx-ai/ogx/pull/7076))
- Smoke suites for the Claude Code client and the agent CLIs ([#6873](https://github.com/ogx-ai/ogx/pull/6873), [#6958](https://github.com/ogx-ai/ogx/pull/6958))

### SystemOne API

- **SystemOne judgment model API** with the TypeSafe provider ([#6748](https://github.com/ogx-ai/ogx/pull/6748))
- Judgment support on **Ollama, llama.cpp, SGLang, and vLLM**, each with its own integration CI lane ([#6769](https://github.com/ogx-ai/ogx/pull/6769), [#6772](https://github.com/ogx-ai/ogx/pull/6772), [#6782](https://github.com/ogx-ai/ogx/pull/6782), [#6866](https://github.com/ogx-ai/ogx/pull/6866), [#6773](https://github.com/ogx-ai/ogx/pull/6773), [#6880](https://github.com/ogx-ai/ogx/pull/6880))

**Also in this release:** Responses robustness (proper 400s for oversized prompts and incomplete `text.format`, malformed tool-call arguments fed back to the model, no more silently dropped turns, `content_filter` → refusal), Vector IO pagination/chunking/embedding fixes, the Simplified Chinese README, and the finalized JOSS manuscript.

## Breaking Changes

### Summary

> **Note:** Each change is described in detail in the sections below.

| Change | Migration | PR |
|--------|-----------|----|
| Python 3.12 no longer supported (3.13–3.14 are) | `ogx`/`ogx-api` wheels require an interpreter ≥ 3.13; Ubuntu 24.04's stock 3.12 needs uv/pyenv | [#6973](https://github.com/ogx-ai/ogx/pull/6973) |
| `tool_runtime` and `tool_groups` APIs removed (incl. `remote::model-context-protocol`) | Web and file search run inside the Responses API; MCP tools are served by the Responses API directly, optionally via MCP connectors | [#7039](https://github.com/ogx-ai/ogx/pull/7039) |
| `remote::passthrough` inference provider removed | Fixed-endpoint case: `remote::openai` with a `base_url` override. Per-request `passthrough_url`/`forward_headers` has **no equivalent** | [#7034](https://github.com/ogx-ai/ogx/pull/7034) |
| `/v1alpha/file-processors/process` route removed | Use the jobs API | [#7078](https://github.com/ogx-ai/ogx/pull/7078) |
| `knowledge_search` compat alias removed (deprecated since v0.7) | Use `file_search` | [#6803](https://github.com/ogx-ai/ogx/pull/6803) |
| `remote::wolfram-alpha` tool provider removed | — | [#6804](https://github.com/ogx-ai/ogx/pull/6804) |
| String forms of `Models.register_model`/`unregister_model` removed | Pass a `RegisterModelRequest`/`UnregisterModelRequest` (or keyword args) | [#7082](https://github.com/ogx-ai/ogx/pull/7082) |
| Stack/distribution configs using the `tool_runtime` API section no longer load | Move search backends to the `web_search` section | [#7079](https://github.com/ogx-ai/ogx/pull/7079) |
| Deprecated CLI shims removed: `ogx letsgo`, `ogx stack run`, `ogx stack go`, `ogx connect codex --exec` | Use `ogx go`, `ogx run`, or `--` forwarding for codex exec | [#7076](https://github.com/ogx-ai/ogx/pull/7076) |
| Deprecated config keys removed: `image_name`, `registered_resources.tool_groups`, vLLM top-level `tls_verify`; `~/.llama` → `~/.ogx` auto-migration gone | Update config files; move any remaining `~/.llama` state manually | [#7077](https://github.com/ogx-ai/ogx/pull/7077) |
| `ogx connect` resolves the server URL once: per-agent `--url`/`--host/--port` flags removed | Use `ogx connect --url ... <agent>` | [#6929](https://github.com/ogx-ai/ogx/pull/6929) |
| Anthropic message batch metadata persisted in `AuthorizedSqlStore` | A SQL store backend must be resolvable at startup for hand-written KV-only messages configs | [#7047](https://github.com/ogx-ai/ogx/pull/7047) |
| `response_format.json_schema.name` required; incomplete `text.format` → 400 at parse time | Send complete `json_schema` formats (`name` + `schema`) | [#6907](https://github.com/ogx-ai/ogx/pull/6907) |

---

### Runtime & platform

#### Python 3.12 dropped ([#6973](https://github.com/ogx-ai/ogx/pull/6973))

**Impact:** Installations on Python 3.12 (e.g. Ubuntu 24.04 stock).

The `ogx` and `ogx-api` wheels now require Python 3.13 or 3.14.

**Migration:** Use an interpreter ≥ 3.13; on systems with stock 3.12, use uv or pyenv to provision one.

### API surface removals

#### `tool_runtime` and `tool_groups` APIs removed ([#7039](https://github.com/ogx-ai/ogx/pull/7039))

**Impact:** Code calling `Api.tool_runtime`/`Api.tool_groups`, the `ToolGroups`/`ToolRuntime`/`ToolStore` protocols, tool-group models, `GET /v1/admin/tools`, or `ogx-client toolgroups`.

Everything is gone, including the `remote::model-context-protocol` provider. Tool-invocation metrics are renamed `ogx.tool_runtime.*` → `ogx.tools.*`.

**Migration:** Function tools are client-side; web and file search run inside the Responses API; MCP tools are served by the Responses API directly, optionally via MCP connectors — nothing functional is lost.

#### `remote::passthrough` inference provider removed ([#7034](https://github.com/ogx-ai/ogx/pull/7034))

**Impact:** Stacks using `remote::passthrough` as an inference provider.

**Migration:** Fixed-endpoint case: use `remote::openai` with a `base_url` override. Per-request `passthrough_url`/`forward_headers` (caller-supplied downstream URL + credentials) has **no equivalent**.

#### `/v1alpha/file-processors/process` route removed ([#7078](https://github.com/ogx-ai/ogx/pull/7078))

**Migration:** Use the jobs API.

#### `knowledge_search` compat alias removed ([#6803](https://github.com/ogx-ai/ogx/pull/6803))

Deprecated since v0.7. **Migration:** Use `file_search`.

#### `remote::wolfram-alpha` tool provider removed ([#6804](https://github.com/ogx-ai/ogx/pull/6804))

#### String forms of `Models.register_model`/`unregister_model` removed ([#7082](https://github.com/ogx-ai/ogx/pull/7082))

**Impact:** Library (not HTTP/SDK) callers passing string arguments.

**Migration:** Pass a `RegisterModelRequest`/`UnregisterModelRequest` (or keyword args).

### Config & CLI

#### `tool_runtime` config section no longer loads ([#7079](https://github.com/ogx-ai/ogx/pull/7079))

**Impact:** Stack configs and distribution configs using the `tool_runtime` API section (including the removed `inline::file-search` provider) **fail to load** — the interim lenient normalizer was itself removed.

**Migration:** Move search backends to the `web_search` section.

#### Deprecated CLI shims removed ([#7076](https://github.com/ogx-ai/ogx/pull/7076))

**Impact:** Scripts using `ogx letsgo`, `ogx stack run`, `ogx stack go`, or `ogx connect codex --exec`.

**Migration:** Use `ogx go` (auto-detect providers and start the stack), `ogx run` for explicit stack configs, or `--` forwarding for codex exec.

#### Deprecated config keys removed ([#7077](https://github.com/ogx-ai/ogx/pull/7077))

**Impact:** Configs using `image_name`, `registered_resources.tool_groups`, or vLLM top-level `tls_verify`; users relying on the `~/.llama` → `~/.ogx` auto-migration.

**Migration:** Update config files; move any remaining `~/.llama` state manually.

#### `ogx connect` resolves the server URL once ([#6929](https://github.com/ogx-ai/ogx/pull/6929))

**Impact:** Scripts using per-agent `--url` (claude/codex) or `--host/--port` (opencode) flags.

**Migration:** Use `ogx connect --url ... <agent>`. URLs without a `/v1` path (or with a non-numeric port) now fail fast with a parse error.

### Behavior

#### Anthropic message batch metadata in `AuthorizedSqlStore` ([#7047](https://github.com/ogx-ai/ogx/pull/7047))

**Impact:** Hand-written KV-only configs for the messages provider.

Batch metadata is now persisted in an `AuthorizedSqlStore` table, so a SQL store backend must be resolvable at startup. Official distributions (all define `sql_default`) are unaffected.

**Migration:** Add a SQL backend to KV-only messages-provider configs.

#### `response_format.json_schema.name` now required ([#6907](https://github.com/ogx-ai/ogx/pull/6907))

**Impact:** Clients sending `json_schema` response formats without a `name`, or incomplete `text.format` payloads.

Incomplete `text.format` now fails at parse time with a 400 (previously a 500 deep in the provider).

**Migration:** Send complete `json_schema` formats (`name` + `schema`).
