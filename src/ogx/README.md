# ogx

Server implementation for OGX. This is the main package that provides the runtime, built-in providers, CLI, and all server-side logic.

## Directory Structure

```text
ogx/
  core/               # Server core: routing, resolution, storage, server
  providers/           # All provider implementations (inline + remote)
  distributions/       # Pre-built distribution configurations
  cli/                 # CLI commands (ogx run, build, configure, etc.)
  models/              # Model metadata and registries
  testing/             # Test infrastructure (API recorder for record/replay)
  telemetry/           # OpenTelemetry integration
  __init__.py
  env.py               # Environment variable utilities
  log.py               # Logging configuration
```

## How It Fits Together

1. **`core/`** handles server startup, provider resolution, request routing, and storage.
2. **`providers/`** contains all provider implementations. Each provider implements a protocol defined in `ogx_api`.
3. **`distributions/`** are pre-configured YAML files that wire together specific providers for common deployment scenarios.
4. **`cli/`** provides the `ogx` command-line tool for running, building, and configuring stacks.

## Key Entry Points

- `core/server/server.py` -- FastAPI application and `main()` entry point
- `core/stack.py` -- `Stack` class that initializes and wires all components
- `core/resolver.py` -- `resolve_impls()` resolves provider dependencies and instantiates them

## Package Relationship

This package (`ogx`) depends on `ogx-api` (`src/ogx_api/`) for protocol definitions and data types. See the root [ARCHITECTURE.md](../../ARCHITECTURE.md) for the full system overview.
