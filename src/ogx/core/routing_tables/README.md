# routing_tables

Resource-to-provider mapping tables. Each routing table tracks which provider owns which resource for a given API.

## Directory Structure

```text
routing_tables/
  __init__.py            # Factory functions: get_routing_table_impl(), get_auto_router_impl()
  common.py              # CommonRoutingTableImpl base class
  models.py              # ModelsRoutingTable (models -> inference providers)
  shields.py             # ShieldsRoutingTable (shields -> safety providers)
  toolgroups.py          # ToolGroupsRoutingTable
  vector_stores.py       # VectorStoresRoutingTable
```

## How Routing Works

Each auto-routed API has a paired routing table API (see `builtin_automatically_routed_apis()` in `core/distribution.py`). For example:

- `Api.models` (routing table) is paired with `Api.inference` (router)
- When a model is registered, the routing table records which provider owns it
- When an inference request arrives, the router asks the routing table for the provider that handles the requested model

## CommonRoutingTableImpl

`CommonRoutingTableImpl` in `common.py` is the base class for all routing tables. It provides:

- Resource registration and unregistration with provider tracking
- Resource lookup by identifier with access control enforcement
- Persistence via `DistributionRegistry` (survives server restarts)
- Provider initialization: on startup, calls each provider's list method to discover pre-existing resources

## Factory Functions

`__init__.py` exports two factory functions:

- `get_routing_table_impl(api, ...)` -- Creates the appropriate routing table for the given API
- `get_auto_router_impl(api, routing_table, ...)` -- Creates the router that uses the routing table to dispatch requests
