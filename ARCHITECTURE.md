# OGX Architecture

This document describes the internal architecture of OGX for contributors and AI agents working with the codebase. For user-facing documentation, see [ogx-ai.github.io](https://ogx-ai.github.io/). For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## System Overview

OGX is a server that exposes a unified API for AI capabilities: inference, agents, safety, vector storage, evaluation, and more. It is provider-agnostic: the same API works whether the backend is Ollama, OpenAI, vLLM, Fireworks, or dozens of other services.

The codebase is split into two packages:

- **`ogx-api`** (`src/ogx_api/`) -- Lightweight package containing API protocol definitions (Python `Protocol` classes), Pydantic data types, and provider spec definitions. No server code, no provider implementations. Third-party providers depend only on this.
- **`ogx`** (`src/ogx/`) -- The server implementation: provider resolution, routing, storage, CLI, and all built-in providers.
- **`ogx-ui`** (`src/ogx_ui/`) -- Optional web UI for the chat playground and admin. Built with Next.js.

## Request Flow

```text
Client (ogx-client SDK or raw HTTP)
  |
  v
FastAPI Server  (src/ogx/core/server/server.py)
  |
  |-- AuthenticationMiddleware  (token validation, user extraction)
  |-- QuotaMiddleware           (rate limiting per client)
  |
  v
Route Dispatch
  |
  |-- FastAPI Router routes     (auto-discovered via fastapi_router_registry.py)
  |
  v
Router  (src/ogx/core/routers/)
  |
  |-- Looks up the resource (model, shield, etc.) in the RoutingTable
  |-- Resolves which provider handles this resource
  |-- Enforces access control policies
  |
  v
Provider Implementation
  |
  |-- Inline provider (runs in-process, e.g. meta-reference, sqlite-vec)
  |-- Remote provider (calls external service, e.g. ollama, openai, fireworks)
  |
  v
External Service or Local Computation
```

### Detailed Flow Example: Chat Completion

1. Client sends `POST /v1/chat/completions` with `model: "ollama/llama3.2:3b-instruct-fp16"`.
2. `server.py` dispatches to the inference FastAPI router.
3. The `InferenceRouter` (`core/routers/inference.py`) calls `routing_table.get_provider_impl(model_id)`.
4. `CommonRoutingTableImpl` looks up the model in `DistributionRegistry`, finds it belongs to provider `ollama`.
5. The router delegates to the `ollama` provider's `openai_chat_completion()` method.
6. The Ollama provider (which extends `OpenAIMixin`) creates an `AsyncOpenAI` client pointing at the Ollama server and forwards the request.
7. The response streams back through the router to the client as SSE events.

## Provider Architecture

### Provider Types

```text
Provider
  |
  |-- InlineProviderSpec    (runs in-process)
  |     provider_type: "inline::builtin"
  |     module: "ogx.providers.inline.inference.builtin"
  |
  |-- RemoteProviderSpec    (adapts an external service)
        provider_type: "remote::ollama"
        module: "ogx.providers.remote.inference.ollama"
```

Each provider spec declares:

- `api` -- which API it implements (e.g., `Api.inference`)
- `provider_type` -- unique identifier like `"remote::openai"`
- `module` -- Python module with a `get_adapter_impl()` or `get_provider_impl()` function
- `config_class` -- Pydantic config model for the provider
- `pip_packages` -- additional dependencies needed at runtime

### Provider Registry

`src/ogx/providers/registry/` contains one file per API (e.g., `inference.py`, `safety.py`). Each file defines an `available_providers()` function that returns all `ProviderSpec` objects for that API. The registry is loaded at startup by `get_provider_registry()` in `core/distribution.py`.

### Provider Resolution

At startup, `resolve_impls()` in `core/resolver.py`:

1. **Validates** providers declared in the run config against the registry.
2. **Sorts** providers by dependency order (e.g., agents depends on inference).
3. **Instantiates** each provider by importing its module and calling its factory function.
4. **Sets up auto-routing**: for APIs like inference, creates a `RoutingTable` + `Router` pair so multiple providers can serve different models through the same API.

### Auto-Routing

Many APIs use automatic routing. For example, `Api.inference` is paired with `Api.models`:

```text
Api.models (RoutingTable)  <-->  Api.inference (Router)
  |                                |
  |-- ModelsRoutingTable           |-- InferenceRouter
  |   tracks which provider        |   delegates to correct
  |   owns which model             |   provider per request
```

The full list of auto-routed pairs is defined in `builtin_automatically_routed_apis()` in `core/distribution.py`:

| Routing Table API     | Router API       |
|-----------------------|------------------|
| `Api.models`          | `Api.inference`  |
| `Api.shields`         | `Api.safety`     |
| `Api.datasets`        | `Api.datasetio`  |
| `Api.scoring_functions` | `Api.scoring`  |
| `Api.benchmarks`      | `Api.eval`       |
| `Api.tool_groups`     | `Api.tool_runtime` |
| `Api.vector_stores`   | `Api.vector_io`  |

## The API Layer (`ogx_api`)

The `ogx_api` package defines all public-facing types and protocols:

- **Protocols** -- Python `Protocol` classes like `Inference`, `Safety` that define the API contract. HTTP routes are defined via FastAPI routers in `fastapi_routes.py` modules.
- **Data Types** -- Pydantic models for requests, responses, and resources (e.g., `Model`, `Shield`, `ChatCompletionRequest`).
- **Provider Specs** -- `InlineProviderSpec`, `RemoteProviderSpec`, and related types that define how providers are declared.
- **Internal utilities** -- KVStore and SqlStore abstract interfaces live here so third-party providers can use them without depending on the full server.

Provider implementations import from `ogx_api` for type definitions and from `ogx.providers.utils` for shared functionality.

## Storage

### Storage Configuration

Storage is configured in the `storage` section of the run config (`StackConfig.storage`). It defines backend references that providers and core services use:

```yaml
storage:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR}/registry.db
  stores:
    kvstore:
      type: kv_sqlite
      db_path: ${env.SQLITE_STORE_DIR}/kvstore.db
    inference:
      type: sql_sqlite
      db_path: ${env.SQLITE_STORE_DIR}/inference_store.db
```

### KVStore

`src/ogx/core/storage/kvstore/` provides a key-value store abstraction (`KVStore`) with backends:

| Backend   | Config Class             | Use Case               |
|-----------|--------------------------|------------------------|
| SQLite    | `SqliteKVStoreConfig`    | Default, single-node   |
| Redis     | `RedisKVStoreConfig`     | Multi-node, caching    |
| PostgreSQL| `PostgresKVStoreConfig`  | Production deployments |
| MongoDB   | `MongoDBKVStoreConfig`   | Document-oriented      |

Used by: distribution registry, quota tracking, provider state.

### SqlStore

`src/ogx/core/storage/sqlstore/` provides a SQL store abstraction (`SqlStore`) with SQLAlchemy backends:

| Backend    | Config Class             | Use Case               |
|------------|--------------------------|------------------------|
| SQLite     | `SqliteSqlStoreConfig`   | Default, single-node   |
| PostgreSQL | `PostgresSqlStoreConfig` | Production deployments |

Used by: inference store (chat completion logs), conversations, prompts.

### Distribution Registry

`src/ogx/core/store/` implements `DistributionRegistry`, which tracks all registered resources (models, shields, datasets, etc.) across providers. It persists to the configured KVStore so resources survive server restarts.

## Configuration

### Run Config (`StackConfig`)

A YAML file that defines everything about a running OGX instance:

```yaml
version: 2
distro_name: starter
apis:
  - inference
  - agents
  - safety
  # ...
providers:
  inference:
    - provider_id: ollama
      provider_type: remote::ollama
      config:
        base_url: ${env.OLLAMA_URL:=http://localhost:11434/v1}
  safety:
    - provider_id: llama-guard
      provider_type: inline::llama-guard
      config: {}
storage:
  type: sqlite
  db_path: ...
```

Key features:

- **Environment variable substitution**: `${env.VAR_NAME:=default}` syntax for config values.
- **Conditional providers**: `${env.API_KEY:+provider_id}` syntax enables a provider only when a variable is set.
- **Multiple providers per API**: e.g., both `ollama` and `openai` can serve inference, each handling different models.

### Distributions

A distribution is a pre-built configuration that bundles specific providers for a target environment. Think of it like Kubernetes distributions (AKS, EKS, GKE): the core API stays the same, but each distribution wires different backends. `src/ogx/distributions/` contains these configurations (e.g., `starter`, `nvidia`). Each distribution directory has:

- `config.yaml` -- the run config
- Templates and codegen support via `template.py`

### Build Config

Used by `ogx build` to create container images. Declares which providers to include and what packages to install. Versioned separately from the run config.

## Recording and Replay Test System

Integration tests use a record/replay system (`src/ogx/testing/api_recorder.py`) that intercepts OpenAI client calls to record real API responses, then replays them for fast, deterministic CI runs.

### How It Works

1. **Recording**: Tests run against a real server. The `APIRecorder` monkey-patches `OpenAI` client methods to capture every request/response pair. Responses are stored as JSON files under `tests/integration/recordings/`.

2. **Replay**: In CI, tests run in replay mode. The recorder matches incoming requests to stored responses by hashing the request parameters, returning cached responses instead of making real API calls.

3. **Modes** (controlled by `--inference-mode` or `OGX_TEST_INFERENCE_MODE`):
   - `replay` (default) -- use cached responses
   - `record` -- force-record all interactions
   - `record-if-missing` -- record only when no cached response exists
   - `live` -- bypass recording entirely, make real calls

4. **Deterministic IDs**: The recorder overrides ID generation (`set_id_override()`) during replay so that resource IDs (files, vector stores, etc.) are reproducible across runs.

### Recording Storage

Recordings live in `tests/integration/recordings/` organized by provider and test. Each recording is a JSON file containing the serialized request parameters and response. An SQLite index maps requests to response files.

For more details, see `tests/README.md` and `tests/integration/README.md`.

## Key Classes and Entry Points

| Component | Location | Purpose |
|-----------|----------|---------|
| `OGX` | `core/stack.py` | Composite class implementing all API protocols |
| `Stack` | `core/stack.py` | Initialization, resource registration, lifecycle |
| `StackApp` | `core/server/server.py` | FastAPI app wrapper |
| `resolve_impls()` | `core/resolver.py` | Provider instantiation and dependency resolution |
| `CommonRoutingTableImpl` | `core/routing_tables/common.py` | Base routing table for all auto-routed APIs |
| `InferenceRouter` | `core/routers/inference.py` | Routes inference calls to correct provider |
| `OpenAIMixin` | `providers/utils/inference/openai_mixin.py` | Shared OpenAI-compatible client logic |
| `get_provider_registry()` | `core/distribution.py` | Loads all available provider specs |
| `APIRecorder` | `testing/api_recorder.py` | Record/replay test infrastructure |

## Directory Map

```text
src/
  ogx_api/          # API definitions package (separate pip package)
    inference/              # Inference protocol, models, FastAPI routes
    responses/              # Responses API protocol and routes
    safety/                 # Safety protocol and routes
    datatypes.py            # Shared data types
    providers/              # Provider spec types
    internal/               # KVStore/SqlStore interfaces
  ogx/              # Server implementation
    core/
      server/               # FastAPI server, auth, routing
      routers/              # API-specific routers (inference, safety, etc.)
      routing_tables/       # Resource-to-provider mapping
      storage/              # KVStore and SqlStore backends
      store/                # Distribution registry
      resolver.py           # Provider resolution engine
      distribution.py       # Provider registry loading
      stack.py              # Stack initialization and lifecycle
    providers/
      inline/               # In-process provider implementations
      remote/               # Remote service adapters
      registry/             # Provider spec declarations
      utils/                # Shared provider utilities
    distributions/          # Pre-built distribution configs
    cli/                    # CLI commands (ogx run, build, etc.)
    testing/                # Test infrastructure (api_recorder)
tests/
  unit/                     # Fast, isolated tests
  integration/              # End-to-end tests with record/replay
    recordings/             # Cached API responses
```
