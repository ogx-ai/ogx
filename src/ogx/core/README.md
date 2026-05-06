# core

Server core for OGX: routing, provider resolution, storage, and the FastAPI server.

## Directory Structure

```text
core/
  server/              # FastAPI server, auth middleware, quota middleware
  routers/             # API-specific routers (inference, safety, eval, etc.)
  routing_tables/      # Resource-to-provider mapping tables
  storage/             # KVStore and SqlStore backends
  store/               # Distribution registry (persists registered resources)
  access_control/      # Access control policy enforcement
  conversations/       # Conversation service (persistence for chat threads)
  prompts/             # Prompt service (prompt template management)
  utils/               # Config resolution, context propagation, dynamic import
  resolver.py          # Provider resolution engine: validate, sort, instantiate
  distribution.py      # Provider registry loading, API enumeration
  stack.py             # Stack class: initialization, resource registration, lifecycle
  datatypes.py         # Core data types (StackConfig, Provider, RoutableObject, etc.)
  library_client.py    # In-process client (no server needed)
  build.py             # Build config handling for container images
  configure.py         # Interactive configuration wizard
```

## Request Lifecycle

1. `server.py` receives an HTTP request and dispatches to the correct handler.
2. The handler calls a **Router** (e.g., `InferenceRouter`) which consults the **RoutingTable** to find the provider for the requested resource.
3. The router delegates to the provider implementation.
4. The provider either computes locally (inline) or calls an external service (remote).

## Key Classes

- `Stack` (`stack.py`) -- Orchestrates initialization: loads config, resolves providers, registers resources, starts background tasks.
- `resolve_impls()` (`resolver.py`) -- The provider resolution engine. Validates providers against the registry, sorts by dependency order, instantiates each one.
- `CommonRoutingTableImpl` (`routing_tables/common.py`) -- Base class for all routing tables. Manages the mapping from resource identifiers to provider implementations.
- `DistributionRegistry` (`store/`) -- Persistent registry of all resources (models, shields, etc.) across providers.
