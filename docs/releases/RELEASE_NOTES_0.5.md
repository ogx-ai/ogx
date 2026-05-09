<!-- markdownlint-disable MD036 -->
# OGX 0.5 Release Notes

**Release Date:** February 2026

Release 0.5 brings significant improvements to API consistency, OpenAI conformance, provider capabilities, and a major architectural refactoring of all APIs to use FastAPI routers. This release also introduces the Connectors API for MCP server management and comprehensive network configuration for remote inference providers.

## Breaking Changes

### Summary

> **Note:** Each change is described in detail with migration instructions in the sections below.

**Hard Breaking Changes (action required before upgrading):**

| Change | Migration | PR |
|--------|-----------|-----|
| Post-Training API endpoints restructured | Update URLs: `/post-training/job/status?job_uuid=X` → `/post-training/jobs/{job_uuid}/status` | [#4606](https://github.com/ogx-ai/ogx/pull/4606) |
| Embeddings API rejects explicit `null` | Remove `dimensions: null` and `user: null` from requests (omit fields instead) | [#4644](https://github.com/ogx-ai/ogx/pull/4644) |
| Safety API provider interface changed | Update provider method signatures to accept `RunShieldRequest` object | [#4643](https://github.com/ogx-ai/ogx/pull/4643) |
| Meta-Reference GPU provider removed | Switch to `remote::vllm` or `remote::ollama` | [#4828](https://github.com/ogx-ai/ogx/pull/4828) |
| Scope-based endpoint authorization removed | Migrate to new YAML-based endpoint authorization | [#4734](https://github.com/ogx-ai/ogx/pull/4734) |

**Deprecated (works with warnings, migrate before next major release):**

| Change | Migration | PR |
|--------|-----------|-----|
| `image_name` → `distro_name` | Replace `image_name:` with `distro_name:` in config files | [#4396](https://github.com/ogx-ai/ogx/pull/4396) |
| Eval API calling convention | Use `RunEvalRequest` objects instead of keyword arguments | [#4425](https://github.com/ogx-ai/ogx/pull/4425) |
| vLLM `tls_verify` field | Move to `network.tls.verify` in provider config | [#4748](https://github.com/ogx-ai/ogx/pull/4748) |

**Behavior Changes (no code changes required, but be aware):**

| Change | Note | PR |
|--------|------|-----|
| `finish_reason` values now OpenAI-conformant | Check if code handles: `stop`, `length`, `tool_calls`, `content_filter` | [#4679](https://github.com/ogx-ai/ogx/pull/4679) |
| Vertex AI defaults to "global" region | Set explicit `region` in config if needed | [#4674](https://github.com/ogx-ai/ogx/pull/4674) |
| Usage token details now always present | No action needed (additive change) | [#4690](https://github.com/ogx-ai/ogx/pull/4690) |

---

### Hard Breaking Changes

These changes take effect immediately and require updates before upgrading.

#### 1. Post-Training API Endpoints Restructured ([#4606](https://github.com/ogx-ai/ogx/pull/4606))

*Contributed by Eoin Fennessy (Red Hat)*

**Impact:** API clients using post-training job endpoints

Post-training job endpoints now use path parameters instead of query parameters for `job_uuid`, following REST best practices:

| Before | After |
|--------|-------|
| `POST /post-training/job/cancel?job_uuid=X` | `POST /post-training/jobs/{job_uuid}/cancel` |
| `GET /post-training/job/status?job_uuid=X` | `GET /post-training/jobs/{job_uuid}/status` |
| `GET /post-training/job/artifacts?job_uuid=X` | `GET /post-training/jobs/{job_uuid}/artifacts` |

**Migration:** Update API calls to use the new URL paths with job UUID in the path.

---

#### 2. Embeddings API Changes for OpenAI Conformance ([#4644](https://github.com/ogx-ai/ogx/pull/4644))

*Contributed by Eoin Fennessy (Red Hat)*

**Impact:** API clients using `/embeddings` endpoint

Several changes to achieve full OpenAI API conformance:

- **`dimensions` and `user` fields now reject explicit `null`**: Previously accepted `dimensions: null`, now returns validation error
- **`encoding_format` is now an enum**: Only `"float"` or `"base64"` allowed

**Before (accepted):**

```python
request = {"model": "test", "input": "hello", "dimensions": None}
```

**After (ValidationError):**

```python
# Must omit the field entirely, not set to null
request = {"model": "test", "input": "hello"}
```

---

#### 3. Safety API Provider Interface Changes ([#4643](https://github.com/ogx-ai/ogx/pull/4643))

*Contributed by Roy Belio (Red Hat)*

**Impact:** Custom safety provider implementations

The Safety API has been migrated to FastAPI routers. Safety providers now receive request objects instead of individual parameters:

**Before:**

```python
async def run_shield(
    self, shield_id: str, messages: list, params: dict
) -> RunShieldResponse: ...
```

**After:**

```python
async def run_shield(self, request: RunShieldRequest) -> RunShieldResponse: ...
```

The `params` field has been removed from `RunShieldRequest` as it was unused.

---

#### 4. Meta-Reference GPU Inference Provider Removed ([#4828](https://github.com/ogx-ai/ogx/pull/4828))

*Contributed by Matthew Farrellee*

**Impact:** Users of the `inline::meta-reference` inference provider

The inline meta-reference GPU inference implementation has been removed. Use `vllm`, `ollama`, or other maintained providers instead.

**Migration:** Switch to `remote::vllm`, `remote::ollama`, or another supported inference provider.

---

#### 5. Scope-Based Endpoint Authorization Removed ([#4734](https://github.com/ogx-ai/ogx/pull/4734))

*Contributed by Derek Higgins (Red Hat)*

**Impact:** Users who configured `required_scope` in custom endpoints

The scope-based authorization feature has been removed. Use the new endpoint authorization with YAML config ([#4448](https://github.com/ogx-ai/ogx/pull/4448)) instead.

---

### Deprecated (With Migration Period)

These changes include backward compatibility. The old behavior continues to work but emits deprecation warnings. Plan to migrate before the next major release.

#### 1. Configuration: `image_name` Renamed to `distro_name` ([#4396](https://github.com/ogx-ai/ogx/pull/4396))

*Contributed by Charlie Doern (Red Hat)*

**Impact:** All existing configuration files (`run.yaml`, `config.yaml`)

The `image_name` field in `StackConfig` has been renamed to `distro_name` to better reflect its purpose.

**Before:**

```yaml
version: 2
image_name: my-stack
```

**After:**

```yaml
version: 2
distro_name: my-stack
```

**Status:** Old `image_name` continues to work and is automatically migrated with a deprecation warning.

---

#### 2. Eval API Provider Interface Changes ([#4425](https://github.com/ogx-ai/ogx/pull/4425), [#4683](https://github.com/ogx-ai/ogx/pull/4683))

*Contributed by Roy Belio (Red Hat), Sai Chandra Pandraju (Red Hat)*

**Impact:** Custom eval provider implementations, direct Eval API callers

The Eval API now uses request objects. Old-style parameter calling is deprecated but supported:

**New style (preferred):**

```python
await eval.run_eval(RunEvalRequest(benchmark_id="...", benchmark_config=...))
```

**Old style (deprecated):**

```python
await eval.run_eval(
    benchmark_id="...", benchmark_config=...
)  # emits DeprecationWarning
```

**Status:** Old calling convention works but emits `DeprecationWarning`. Backward compatibility layer added in [#4683](https://github.com/ogx-ai/ogx/pull/4683).

---

#### 3. vLLM `tls_verify` Configuration Deprecated ([#4748](https://github.com/ogx-ai/ogx/pull/4748))

*Contributed by Sébastien Han (Red Hat)*

**Impact:** vLLM provider users with TLS configuration

The `tls_verify` field is deprecated in favor of the new `network.tls.verify` configuration:

**Before:**

```yaml
providers:
  inference:
    - provider_type: remote::vllm
      config:
        url: https://vllm-server
        tls_verify: /path/to/ca.crt
```

**After:**

```yaml
providers:
  inference:
    - provider_type: remote::vllm
      config:
        url: https://vllm-server
        network:
          tls:
            verify: /path/to/ca.crt
```

**Status:** Old `tls_verify` is automatically migrated with a deprecation warning.

---

### Behavior Changes

These are changes to default values or response formats. Existing code should continue to work, but behavior may differ.

#### 1. `finish_reason` Values Changed for OpenAI Conformance ([#4679](https://github.com/ogx-ai/ogx/pull/4679))

*Contributed by Sébastien Han (Red Hat)*

The `finish_reason` field in inference responses now conforms to the OpenAI specification. If your code explicitly checks for specific `finish_reason` values, verify it handles the standard values: `stop`, `length`, `tool_calls`, `content_filter`, `function_call`.

#### 2. Vertex AI Default Region Changed ([#4674](https://github.com/ogx-ai/ogx/pull/4674))

*Contributed by Ken Dreyer (Red Hat)*

The Vertex AI provider now defaults to the "global" OpenAI API endpoint instead of a specific GCP region. If you relied on the previous default region, explicitly configure it in your provider settings.

#### 3. Usage Token Details Now Always Present ([#4690](https://github.com/ogx-ai/ogx/pull/4690))

*Contributed by Matthew Farrellee*

The `input_tokens_details` and `output_tokens_details` fields in the `Usage` object are now always present (previously optional). This is backwards compatible for consumers.

---

## New Features

### Connectors API ([#4263](https://github.com/ogx-ai/ogx/pull/4263), [#4402](https://github.com/ogx-ai/ogx/pull/4402), [#4760](https://github.com/ogx-ai/ogx/pull/4760))

*Contributed by Jaideep Rao (Red Hat)*

New API for managing MCP (Model Context Protocol) server connections via static configuration:

```yaml
connectors:
  - connector_id: kubernetes
    url: "http://localhost:8080/mcp"
    connector_type: mcp
```

API endpoints:

- `GET /v1alpha/connectors` - List all connectors
- `GET /v1alpha/connectors/{connector_id}` - Get connector details
- `GET /v1alpha/connectors/{connector_id}/tools/{tool_name}` - Get tool info

### Comprehensive Network Configuration ([#4748](https://github.com/ogx-ai/ogx/pull/4748))

*Contributed by Sébastien Han (Red Hat)*

All remote inference providers now support unified network configuration for TLS (including mTLS), proxy servers, timeouts, and custom headers:

```yaml
providers:
  inference:
    - provider_type: remote::openai
      config:
        network:
          tls:
            verify: true
            ca_cert: /path/to/ca.crt
            client_cert: /path/to/client.crt
            client_key: /path/to/client.key
          proxy:
            url: http://proxy:8080
          timeout:
            connect: 10.0
            read: 60.0
          headers:
            X-Custom-Header: value
```

### Endpoint Authorization with YAML Config ([#4448](https://github.com/ogx-ai/ogx/pull/4448))

*Contributed by Derek Higgins (Red Hat)*

Infrastructure-level access control for API endpoints:

```yaml
server:
  auth:
    endpoint_policy:
      - path: "/v1/files*"
        conditions:
          - role: admin
      - path: "/v1/health"
        allow: true
```

### Rerankers Support ([#4456](https://github.com/ogx-ai/ogx/pull/4456))

*Contributed by Varsha (Red Hat)*

Vector stores now support reranking for hybrid search:

```python
results = await client.vector_stores.search(
    vector_store_id="vs_123",
    query="search query",
    search_type="hybrid",
    reranker_type="reciprocal_rank_fusion",
    reranker_params={"k": 60},
)
```

### Response API Enhancements

- **`reasoning.effort` parameter** ([#4633](https://github.com/ogx-ai/ogx/pull/4633)) - *Nehanth Narendrula*: Control reasoning token usage

  ```python
  response = client.responses.create(
      model="openai/gpt-5",
      reasoning={"effort": "high"},
      input=[{"role": "user", "content": "Complex problem..."}],
  )
  ```

- **`max_output_tokens`** ([#4592](https://github.com/ogx-ai/ogx/pull/4592)) - *Guangya Liu*: Limit response length

- **`parallel_tool_calls`** ([#4608](https://github.com/ogx-ai/ogx/pull/4608)) - *Shabana Baig (Red Hat)*: Enable parallel tool execution

- **`safety_identifier`** ([#4793](https://github.com/ogx-ai/ogx/pull/4793)) - *Guangya Liu*: Custom safety monitoring tracking

### New Providers

- **Elasticsearch Vector IO** ([#4007](https://github.com/ogx-ai/ogx/pull/4007)) - *Enrico Zimuel (Elastic)*: Use Elasticsearch as a vector store backend
- **OCI 26ai Vector Support** ([#4411](https://github.com/ogx-ai/ogx/pull/4411)) - *Robert Riley (Oracle)*: Oracle Cloud Infrastructure 26ai as vector store

### PGVector Improvements

*All PGVector improvements contributed by Ian Miller (Red Hat)*

- **HNSW indexes** ([#4696](https://github.com/ogx-ai/ogx/pull/4696)): Better vector search performance
- **IVFFlat indexes** ([#4772](https://github.com/ogx-ai/ogx/pull/4772)): Alternative indexing strategy
- **Configurable distance metrics** ([#4714](https://github.com/ogx-ai/ogx/pull/4714)): cosine, euclidean, inner_product
- **Embedding dimension validation** ([#4732](https://github.com/ogx-ai/ogx/pull/4732))
- **Automatic vector extension creation** ([#4660](https://github.com/ogx-ai/ogx/pull/4660))

### Library Client Improvements

- **Shutdown functionality** ([#4642](https://github.com/ogx-ai/ogx/pull/4642)) - *Sergey Yedrikov (Red Hat)*: Proper cleanup for `OGXAsLibraryClient` and `AsyncOGXAsLibraryClient`

  ```python
  async with AsyncOGXAsLibraryClient(config_path) as client:
      # Use client
      ...
  # Automatic cleanup on exit
  ```

### Safety API Improvements

- **`run_moderation` for all providers** ([#4662](https://github.com/ogx-ai/ogx/pull/4662)) - *Mac Misiura (Red Hat)*: OpenAI-compatible moderation API support across NVIDIA, Bedrock, SambaNova, and PromptGuard providers

### ARM64 Support

- **ARM64-based UBI starter image** ([#4474](https://github.com/ogx-ai/ogx/pull/4474)) - *Doug Edgar (Red Hat)*: Container images now available for ARM64 architecture

## API Migration to FastAPI Routers

All APIs have been migrated from the legacy `@webmethod` decorator pattern to FastAPI routers:

- Inference API ([#4755](https://github.com/ogx-ai/ogx/pull/4755)) - *Roy Belio (Red Hat)*
- Agents/Responses API ([#4376](https://github.com/ogx-ai/ogx/pull/4376)) - *Sumanth Kamenani*
- Safety API ([#4643](https://github.com/ogx-ai/ogx/pull/4643)) - *Roy Belio (Red Hat)*
- Eval API ([#4425](https://github.com/ogx-ai/ogx/pull/4425)) - *Roy Belio (Red Hat)*
- Vector IO API ([#4595](https://github.com/ogx-ai/ogx/pull/4595)) - *Sumanth Kamenani*
- Conversations API ([#4342](https://github.com/ogx-ai/ogx/pull/4342)) - *Sébastien Han (Red Hat)*
- Models API ([#4407](https://github.com/ogx-ai/ogx/pull/4407)) - *Nathan Weinberg (Red Hat)*
- Shields API ([#4412](https://github.com/ogx-ai/ogx/pull/4412)) - *Nathan Weinberg (Red Hat)*
- DatasetIO API ([#4400](https://github.com/ogx-ai/ogx/pull/4400)) - *Nathan Weinberg (Red Hat)*
- Prompts API ([#4649](https://github.com/ogx-ai/ogx/pull/4649)) - *Nathan Weinberg (Red Hat)*
- Scoring API ([#4521](https://github.com/ogx-ai/ogx/pull/4521)) - *Guangya Liu*
- Scoring Functions API ([#4599](https://github.com/ogx-ai/ogx/pull/4599)) - *Eleanor (Red Hat)*
- Post-Training API ([#4496](https://github.com/ogx-ai/ogx/pull/4496)) - *Eoin Fennessy (Red Hat)*
- Connectors API ([#4402](https://github.com/ogx-ai/ogx/pull/4402)) - *Jaideep Rao (Red Hat)*

This provides better OpenAPI documentation, improved request validation, and consistent error handling.

## Backward Compatibility Improvements

Several providers have been updated to handle legacy data formats:

- **FAISS** ([#4463](https://github.com/ogx-ai/ogx/pull/4463)) - *Sébastien Han (Red Hat)*: Handles legacy `EmbeddedChunk` format
- **PGVector** ([#4506](https://github.com/ogx-ai/ogx/pull/4506)) - *Ignas Baranauskas (Red Hat)*: Handles legacy chunk format
- **Qdrant** ([#4495](https://github.com/ogx-ai/ogx/pull/4495)) - *Ignas Baranauskas (Red Hat)*: Handles legacy chunk format
- **Milvus** ([#4484](https://github.com/ogx-ai/ogx/pull/4484)) - *Francisco Javier Arceo (Red Hat)*: Handles legacy chunk format
- **SQLite-vec, Chroma, Weaviate** ([#4502](https://github.com/ogx-ai/ogx/pull/4502)) - *Christian Zaccaria (Red Hat)*: Handle legacy chunk formats

## Bug Fixes

- MCP CPU spike fixed by using context manager for session cleanup ([#4758](https://github.com/ogx-ai/ogx/pull/4758)) - *Bill Murdock (Red Hat)*
- Concurrent SentenceTransformer loading race condition fixed ([#4636](https://github.com/ogx-ai/ogx/pull/4636)) - *Sergey Yedrikov (Red Hat)*
- Reasoning content field compatibility with Ollama and vLLM ([#4715](https://github.com/ogx-ai/ogx/pull/4715)) - *Shabana Baig (Red Hat)*
- File search results now include document attributes/metadata ([#4680](https://github.com/ogx-ai/ogx/pull/4680)) - *Cesare Pompeiano (Red Hat)*
- Vector store registration from config with OpenAI metadata ([#4616](https://github.com/ogx-ai/ogx/pull/4616)) - *Cesare Pompeiano (Red Hat)*
- Session polling enabled during streaming responses ([#4738](https://github.com/ogx-ai/ogx/pull/4738)) - *Roland Huß (Red Hat)*
- Security vulnerabilities in GitHub Actions workflows addressed ([#4752](https://github.com/ogx-ai/ogx/pull/4752)) - *Charlie Doern (Red Hat)*

## Documentation

- Updated quick start guide ([#4435](https://github.com/ogx-ai/ogx/pull/4435)) - *Guangya Liu*
- Migration guide from Agents to Responses API ([#4375](https://github.com/ogx-ai/ogx/pull/4375)) - *Bill Murdock (Red Hat)*
- Contributing guidelines for integration tests ([#4460](https://github.com/ogx-ai/ogx/pull/4460)) - *Guangya Liu*
- Provider contribution guide ([#4478](https://github.com/ogx-ai/ogx/pull/4478)) - *Sébastien Han (Red Hat)*
- Release process documentation ([#4470](https://github.com/ogx-ai/ogx/pull/4470)) - *Raghotham Murthy*

## Upgrade Guide

### Before Upgrading (Hard Breaking Changes)

Complete these steps before upgrading to 0.5:

1. **Search for post-training API calls** in your codebase:

   ```bash
   grep -r "post-training/job" .
   ```

   Update any matches to use the new path format: `/post-training/jobs/{job_uuid}/...`

2. **Search for null values in embeddings requests**:

   ```bash
   grep -rE '"(dimensions|user)":\s*null' .
   ```

   Remove these fields entirely instead of setting them to `null`.

3. **If using `inline::meta-reference` provider**, update your config:

   ```yaml
   # Before
   provider_type: inline::meta-reference

   # After (choose one)
   provider_type: remote::vllm
   provider_type: remote::ollama
   ```

4. **If you have custom safety providers**, update method signatures:

   ```python
   # Before
   async def run_shield(
       self, shield_id: str, messages: list, params: dict
   ) -> RunShieldResponse: ...


   # After
   async def run_shield(self, request: RunShieldRequest) -> RunShieldResponse: ...
   ```

5. **If using `required_scope` for endpoint auth**, migrate to new YAML config:

   ```yaml
   server:
     auth:
       endpoint_policy:
         - path: "/v1/inference/*"
           conditions:
             - role: inference-user
   ```

6. **Regenerate client SDKs** if you use generated clients from OpenAPI specs.

### After Upgrading (Deprecations)

Address these warnings before the next major release:

1. **Update configuration files**: Replace `image_name` with `distro_name`
2. **Update vLLM TLS config**: Move `tls_verify` to `network.tls.verify`
3. **Update Eval API calls**: Use `RunEvalRequest` objects instead of keyword arguments
