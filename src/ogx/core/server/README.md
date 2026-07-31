# server

FastAPI server implementation for OGX.

## Directory Structure

```text
server/
  __init__.py
  server.py                    # Main FastAPI app, route dispatch, SSE streaming, lifespan
  auth.py                      # AuthenticationMiddleware (Bearer token validation)
  auth_providers.py            # Auth provider implementations (Kubernetes, custom endpoint)
  metrics.py                   # RequestMetricsMiddleware (per-API request metrics)
  routes.py                    # Route initialization and matching from FastAPI routers
  fastapi_router_registry.py   # Auto-discovery of FastAPI routers from ogx_api packages
```

## How It Works

### Server Startup

1. `main()` in `server.py` resolves the config, creates a `StackApp` (subclass of `FastAPI`).
2. The lifespan context awaits `stack.initialize()` (provider resolution, resource registration), then registers routers.
3. The lifespan starts background tasks (e.g., periodic registry refresh).

### Route Registration

Routes are defined as native FastAPI routers. `fastapi_router_registry.py` auto-discovers router factories by scanning `ogx_api.<api>.fastapi_routes` modules for `create_router` functions. During lifespan startup, `server.py` calls `build_fastapi_router()` for each enabled API and includes the resulting router in the FastAPI app. External APIs can also register router factories via `register_external_api_routers()`.

### Middleware

- **`RequestMetricsMiddleware`** (`metrics.py`): Tracks per-API request counts and latency metrics. Runs as the outermost middleware.
- **`AuthenticationMiddleware`** (`auth.py`): Validates Bearer tokens using a configured auth provider (Kubernetes, custom endpoint). Extracts user identity, attributes, and `tenant_id` for access control. Each auth provider resolves `tenant_id` from its source (JWT claim, HTTP header, K8s claim, or custom endpoint field). Endpoints can opt out by setting `openapi_extra={PUBLIC_ROUTE_KEY: True}` on their route.
- **`TenancyMiddleware`** (`auth.py`): Enforces the configured tenancy mode after authentication. In `disabled` mode: passthrough. In `single` mode: overrides `tenant_id` to the configured default (works with or without auth). In `multi` mode: rejects requests with no `tenant_id` (401).
- **`RouteAuthorizationMiddleware`** (`auth.py`): Enforces route-level access policies based on user roles.
- **`ClientVersionMiddleware`** (`server.py`): Rejects requests from clients with incompatible major.minor versions.
- **`ProviderDataMiddleware`** (`server.py`): Sets up request context for provider data propagation and test context.

### Metrics Export

OTel metrics can be exported two ways, independently and simultaneously:

- **OTLP push** — set `OTEL_EXPORTER_OTLP_ENDPOINT` to push metrics to an OTel Collector.
- **Metrics scrape endpoint** — set `OGX_METRICS_ENDPOINT_ENABLED` (`1`/`true`/`yes`/`on`) to
  expose all metrics in Prometheus exposition format, suitable for scrape-based monitoring
  systems. Metrics are served by a standalone HTTP server on a dedicated port
  (`OGX_METRICS_PORT`, default `9464`; bind address `OGX_METRICS_HOST`, default `127.0.0.1`),
  separate from the main API. It binds to loopback by default; set `OGX_METRICS_HOST` (e.g.
  `0.0.0.0`) to expose it to other hosts or pods. Keeping the scrape endpoint off the API
  port means collectors need no API authentication and the metrics are not reachable by
  regular API consumers.

Telemetry is configured by `ogx.telemetry.initialize_telemetry()`, called from
`Stack.initialize()` (server and library modes). It does not run at import, so non-serving
commands (e.g. `ogx stack list-deps`) neither configure telemetry nor open a network port.
When ogx is launched under `opentelemetry-instrument`, the auto-instrumentation owns the
global `MeterProvider` and manages OTLP export; in that case the scrape reader is added to
the existing provider instead of installing a competing one, so ogx metrics still reach the
scrape endpoint.

### Response Handling

- Non-streaming responses return JSON via FastAPI's standard response handling.
- Streaming responses use Server-Sent Events (SSE) via `StreamingResponse`, with `create_sse_event()` serializing each chunk.
- Exceptions are translated to appropriate HTTP status codes by `translate_exception()`.
