# server

FastAPI server implementation for OGX.

## Directory Structure

```text
server/
  __init__.py
  server.py                    # Main FastAPI app, route dispatch, SSE streaming, lifespan
  auth.py                      # AuthenticationMiddleware (Bearer token validation)
  auth_providers.py            # Auth provider implementations (Kubernetes, custom endpoint)
  quota.py                     # QuotaMiddleware (rate limiting per client)
  metrics.py                   # RequestMetricsMiddleware (per-API request metrics)
  routes.py                    # Route initialization and matching from FastAPI routers
  fastapi_router_registry.py   # Auto-discovery of FastAPI routers from ogx_api packages
```

## How It Works

### Server Startup

1. `main()` in `server.py` resolves the config, creates a `StackApp` (subclass of `FastAPI`).
2. `StackApp.__init__` creates and initializes a `Stack` instance (provider resolution, resource registration).
3. The lifespan context starts background tasks (e.g., periodic registry refresh).

### Route Registration

Routes are defined as native FastAPI routers. `fastapi_router_registry.py` auto-discovers router factories by scanning `ogx_api.<api>.fastapi_routes` modules for `create_router` functions. At startup, `server.py` calls `build_fastapi_router()` for each enabled API and includes the resulting router in the FastAPI app. External APIs can also register router factories via `register_external_api_routers()`.

### Middleware

- **`RequestMetricsMiddleware`** (`metrics.py`): Tracks per-API request counts and latency metrics. Runs as the outermost middleware.
- **`AuthenticationMiddleware`** (`auth.py`): Validates Bearer tokens using a configured auth provider (Kubernetes, custom endpoint). Extracts user identity and attributes for access control. Endpoints can opt out by setting `openapi_extra={PUBLIC_ROUTE_KEY: True}` on their route.
- **`RouteAuthorizationMiddleware`** (`auth.py`): Enforces route-level access policies based on user roles.
- **`QuotaMiddleware`** (`quota.py`): Enforces per-client rate limits (separate limits for authenticated vs. anonymous). Uses KVStore for tracking request counts.
- **`ClientVersionMiddleware`** (`server.py`): Rejects requests from clients with incompatible major.minor versions.
- **`ProviderDataMiddleware`** (`server.py`): Sets up request context for provider data propagation and test context.

### Response Handling

- Non-streaming responses return JSON via FastAPI's standard response handling.
- Streaming responses use Server-Sent Events (SSE) via `StreamingResponse`, with `create_sse_event()` serializing each chunk.
- Exceptions are translated to appropriate HTTP status codes by `translate_exception()`.
