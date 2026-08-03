# Neo4j VectorIO Provider Implementation Plan

Goal: add a first-slice `remote::neo4j` VectorIO provider that behaves like the
other OGX vector store providers and proves deterministic graph-aware retrieval
without changing the public OGX API.

## Review Findings

- The provider must use Neo4j's async driver internally because `VectorIO` is
  async. The live integration test should therefore use that driver through the
  real adapter contract, rather than relying on the synchronous OGX client
  wrapper or a mocked driver.
- It tried to pass graph-specific params through `EmbeddingIndex` methods, whose
  shared protocol does not accept backend-specific params. The corrected design
  performs graph expansion in the Neo4j adapter after normal query execution.
- It proposed enabling shared filter and OpenAI vector-store allowlists before
  proving Neo4j compatibility. The first implementation should explicitly leave
  those broad allowlists unchanged unless the focused tests pass.
- It mixed a full generated-docs rollout into the first code slice. Generated
  provider docs remain part of the provider change, while CI is included here
  because the deterministic Neo4j integration test must run against a real
  service in pull requests.
- The deterministic graph test should not depend on a sentence-transformer query
  embedding matching hand-written 3D chunk embeddings. It should use fixed
  vectors and deterministic keyword input against a local Neo4j container while
  exercising `Neo4jVectorIOAdapter`; direct index calls may be used only for the
  focused vector-index assertion.

## Scope

In scope:

- `Neo4jVectorIOConfig` with documented Pydantic fields and sample env config.
- `remote::neo4j` registry entry and `neo4j` dependency wiring.
- Provider factory, driver lifecycle, KV-backed vector-store registration, chunk
  insert/delete, vector search, keyword search, hybrid search, and graph expansion.
- Unit tests for provider-local config, lifecycle, filter translation, hybrid
  scoring, and graph-expansion ranking. These tests must not connect to Neo4j.
- A focused optional integration test that runs only when `NEO4J_URI` is set and
  exercises the real adapter, index creation, chunk persistence, retrieval, and
  deterministic relationships against a live Neo4j process. It compares the
  same query with graph retrieval disabled and enabled to demonstrate the
  additional related chunk returned by GraphRAG.

Out of scope for this slice:

- New public API fields.
- Router or shared `EmbeddingIndex` protocol changes.
- Adding Neo4j to broad filter/OpenAI vector-store allowlists.
- Broader provider allowlist changes and unrelated CI suites.
- The VectorIO GitHub Actions matrix is in scope: it starts Neo4j 2025.10,
  waits for Bolt readiness, passes connection variables to the Neo4j job, and
  uploads Neo4j logs on failure.

## Design

### Provider Config

Create `src/ogx/providers/remote/vector_io/neo4j/config.py` with
`Neo4jVectorIOConfig`.

Required fields:

- `uri`, default `bolt://localhost:7687`
- `user`, default `neo4j`
- `password`, optional `SecretStr`
- `database`, default `neo4j`
- `index_prefix`, default `ogx`
- `graph_retrieval_enabled`, default `False`
- `graph_expansion_depth`, default `1`, allowed range `1..3`
- `graph_max_neighbors`, default `10`
- `graph_expansion_weight`, default `0.15`
- `graph_relationship_types`, optional list of Neo4j relationship types
- `persistence: KVStoreReference`
- `metadata_store: SqlStoreReference | None`

`sample_run_config()` should use `${env.NEO4J_*}` templates and persist under
`vector_io::neo4j`.

### Adapter

Create `src/ogx/providers/remote/vector_io/neo4j/__init__.py` and
`src/ogx/providers/remote/vector_io/neo4j/neo4j.py`.

The adapter should:

- Extend `OpenAIVectorStoreMixin`, `VectorIO`, and `VectorStoresProtocolPrivate`.
- Use `AsyncGraphDatabase.driver()` and verify connectivity during initialize.
- Initialize `kvstore` from `config.persistence`.
- Initialize optional `metadata_store` using `authorized_sqlstore()`.
- Hydrate persisted vector stores from KV under `vector_stores:neo4j:v1::`.
- Implement `register_vector_store`, `unregister_vector_store`,
  `_get_and_cache_vector_store_index`, `insert_chunks`, `query_chunks`, and
  `delete_chunks`.
- Call `initialize_openai_vector_stores()` after KV/metadata store setup.
- Close the driver on shutdown.

`query_chunks()` should call the shared `VectorStoreWithIndex.query_chunks()`,
then call a provider-local graph expansion helper when either provider config or
request params enable graph retrieval.

### Neo4jIndex

`Neo4jIndex` should implement `EmbeddingIndex`.

Node model:

- One per-vector-store chunk label, derived from a sanitized vector-store id.
- Properties:
  - `chunk_id`
  - `content`
  - `embedding`
- `chunk_content`, stored as a JSON string
- safe metadata mirror properties named `metadata_<key>` for primitive metadata
  values used by filters
- `vector_store_id`

Chunk writes are upserts keyed by `chunk_id`. They replace all provider-owned
node properties, including mirrored metadata, so re-ingesting a chunk cannot
leave stale filterable metadata behind. Replacing properties must preserve
relationships attached to the chunk node.

Indexes:

- Neo4j uniqueness constraint on `chunk_id` for each vector-store label.
- Neo4j vector index on `embedding`.
- Neo4j full-text index on `content`.

Queries:

- Vector: `CALL db.index.vector.queryNodes(...)`.
- Keyword: `CALL db.index.fulltext.queryNodes(...)`.
- Hybrid: query vector and keyword separately, combine using
  `WeightedInMemoryAggregator.combine_search_results()`.

### Filters

Support metadata filters in the first slice because broad VectorIO tests already
exercise provider filters. Implement `_translate_filters()` for:

- comparison: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `nin`
- compound: `and`, `or`

Validate metadata keys with a conservative identifier regexp and use Cypher
parameters for all values. Translate metadata filters against the mirrored
`metadata_<key>` properties, not a nested `metadata` map, because Neo4j node
properties cannot be nested maps. Unsupported filters should raise
`NotImplementedError` with a `Failed to ...` message.

### Graph Expansion

Graph expansion remains provider-local and runs after normal retrieval.

Inputs:

- seed chunks and scores from normal query execution
- `graph_retrieval_enabled`
- `graph_expansion_depth`
- `graph_max_neighbors`
- `graph_expansion_weight`
- optional `graph_relationship_types`

Behavior:

- If disabled or there are no seed chunks, return the original response.
- Traverse from seed chunk nodes to neighboring chunk nodes through graph
  relationship paths.
- Do not return the seed chunk as its own neighbor.
- Score neighbor chunks as `max(seed_score) * graph_expansion_weight`.
- Merge seed and neighbor chunks, preserve the highest score per chunk id, and
  return top `max_chunks`.

For the first implementation, graph fixtures can manually create relationships
from chunk nodes to `(:Entity)` nodes in tests.

### Registration And Dependencies

- Add `RemoteProviderSpec` for `remote::neo4j` in
  `src/ogx/providers/registry/vector_io.py`.
- Add `neo4j` to `starter`, `test`, and `type_checking` dependency surfaces in
  `pyproject.toml`.
- Do not update generated provider docs or CI matrix until the focused unit and
  optional local integration tests pass.

## TDD Checklist

1. Add failing unit tests in `tests/unit/providers/vector_io/test_neo4j.py`.
2. Verify they fail because the module/provider is missing.
3. Implement config, factory, and lifecycle.
4. Verify unit tests pass.
5. Add failing unit tests for filter translation and graph expansion helpers.
6. Implement filters and graph expansion.
7. Verify unit tests pass.
8. Add optional focused integration test in
   `tests/integration/vector_io/test_neo4j_graph_retrieval.py`. This test must
   use `Neo4jVectorIOAdapter`, not only a direct in-process `Neo4jIndex`, so the
   registered provider path is covered against a real Neo4j process.
9. Run it with local Neo4j when available:

```bash
NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=ogxpassword \
uv run pytest tests/integration/vector_io/test_neo4j_graph_retrieval.py -q
```

1. Run focused style/pre-commit checks for changed files.

## Local Neo4j

Known-good local container:

```bash
docker run -d \
  --name ogx-neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/ogxpassword \
  -v ogx-neo4j-data:/data \
  neo4j:2025.10
```

Readiness check:

```bash
for i in {1..60}; do
  if docker exec ogx-neo4j cypher-shell -u neo4j -p ogxpassword 'RETURN 1 AS ok'; then
    exit 0
  fi
  sleep 2
done
docker logs ogx-neo4j
exit 1
```
