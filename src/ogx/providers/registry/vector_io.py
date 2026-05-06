# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
    RemoteProviderSpec,
)

# Common dependencies for all vector IO providers that support document processing
DEFAULT_VECTOR_IO_DEPS = ["chardet", "pypdf>=6.10.0"]


def available_providers() -> list[ProviderSpec]:
    """Return the list of available vector I/O provider specifications.

    Returns:
        List of ProviderSpec objects describing available providers
    """
    return [
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::faiss",
            pip_packages=["faiss-cpu"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.inline.vector_io.faiss",
            config_class="ogx.providers.inline.vector_io.faiss.FaissVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
[Faiss](https://github.com/facebookresearch/faiss) is an inline vector database provider for OGX. It
allows you to store and query vectors directly in memory.
That means you'll get fast and efficient vector retrieval.

## Features

- Lightweight and easy to use
- Fully integrated with OGX
- GPU support
- **Vector search** - FAISS supports pure vector similarity search using embeddings

## Search Modes

**Supported:**
- **Vector Search** (`mode="vector"`): Performs vector similarity search using embeddings

**Not Supported:**
- **Keyword Search** (`mode="keyword"`): Not supported by FAISS
- **Hybrid Search** (`mode="hybrid"`): Not supported by FAISS

> **Note**: FAISS is designed as a pure vector similarity search library. See the [FAISS GitHub repository](https://github.com/facebookresearch/faiss) for more details about FAISS's core functionality.

## Usage

To use Faiss in your OGX project, follow these steps:

1. Install the necessary dependencies.
2. Configure your OGX project to use Faiss.
3. Start storing and querying vectors.

## Installation

You can install Faiss using pip:

```bash
pip install faiss-cpu
```
## Documentation
See [Faiss' documentation](https://faiss.ai/) or the [Faiss Wiki](https://github.com/facebookresearch/faiss/wiki) for
more details about Faiss in general.
""",
        ),
        # NOTE: sqlite-vec cannot be bundled into the container image because it does not have a
        # source distribution and the wheels are not available for all platforms.
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::sqlite-vec",
            pip_packages=["sqlite-vec"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.inline.vector_io.sqlite_vec",
            config_class="ogx.providers.inline.vector_io.sqlite_vec.SQLiteVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
[SQLite-Vec](https://github.com/asg017/sqlite-vec) is an inline vector database provider for OGX. It
allows you to store and query vectors directly within an SQLite database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features

- Lightweight and easy to use
- Fully integrated with OGXs
- Uses disk-based storage for persistence, allowing for larger vector storage

### Comparison to Faiss

The choice between Faiss and sqlite-vec should be made based on the needs of your application,
as they have different strengths.

#### Choosing the Right Provider

Scenario | Recommended Tool | Reason
-- |-----------------| --
Online Analytical Processing (OLAP) | Faiss           | Fast, in-memory searches
Online Transaction Processing (OLTP) | sqlite-vec      | Frequent writes and reads
Frequent writes | sqlite-vec      | Efficient disk-based storage and incremental indexing
Large datasets | sqlite-vec      | Disk-based storage for larger vector storage
Datasets that can fit in memory, frequent reads | Faiss | Optimized for speed, indexing, and GPU acceleration

#### Empirical Example

Consider the histogram below in which 10,000 randomly generated strings were inserted
in batches of 100 into both Faiss and sqlite-vec using `client.tool_runtime.rag_tool.insert()`.

```{image} ../../../../_static/providers/vector_io/write_time_comparison_sqlite-vec-faiss.png
:alt: Comparison of SQLite-Vec and Faiss write times
:width: 400px
```

You will notice that the average write time for `sqlite-vec` was 788ms, compared to
47,640ms for Faiss. While the number is jarring, if you look at the distribution, you can see that it is rather
uniformly spread across the [1500, 100000] interval.

Looking at each individual write in the order that the documents are inserted you'll see the increase in
write speed as Faiss reindexes the vectors after each write.
```{image} ../../../../_static/providers/vector_io/write_time_sequence_sqlite-vec-faiss.png
:alt: Comparison of SQLite-Vec and Faiss write times
:width: 400px
```

In comparison, the read times for Faiss was on average 10% faster than sqlite-vec.
The modes of the two distributions highlight the differences much further where Faiss
will likely yield faster read performance.

```{image} ../../../../_static/providers/vector_io/read_time_comparison_sqlite-vec-faiss.png
:alt: Comparison of SQLite-Vec and Faiss read times
:width: 400px
```

## Usage

To use sqlite-vec in your OGX project, follow these steps:

1. Install the necessary dependencies.
2. Configure your OGX project to use SQLite-Vec.
3. Start storing and querying vectors.

The SQLite-vec provider supports three search modes:

1. **Vector Search** (`mode="vector"`): Performs pure vector similarity search using the embeddings.
2. **Keyword Search** (`mode="keyword"`): Performs full-text search using SQLite's FTS5.
3. **Hybrid Search** (`mode="hybrid"`): Combines both vector and keyword search for better results. First performs keyword search to get candidate matches, then applies vector similarity search on those candidates.

Example with hybrid search:
```python
response = await vector_io.query_chunks(
    vector_store_id="my_db",
    query="your query here",
    params={"mode": "hybrid", "max_chunks": 3, "score_threshold": 0.7},
)

# Using RRF ranker
response = await vector_io.query_chunks(
    vector_store_id="my_db",
    query="your query here",
    params={
        "mode": "hybrid",
        "max_chunks": 3,
        "score_threshold": 0.7,
        "ranker": {"type": "rrf", "impact_factor": 60.0},
    },
)

# Using weighted ranker
response = await vector_io.query_chunks(
    vector_store_id="my_db",
    query="your query here",
    params={
        "mode": "hybrid",
        "max_chunks": 3,
        "score_threshold": 0.7,
        "ranker": {"type": "weighted", "alpha": 0.7},  # 70% vector, 30% keyword
    },
)
```

Example with explicit vector search:
```python
response = await vector_io.query_chunks(
    vector_store_id="my_db",
    query="your query here",
    params={"mode": "vector", "max_chunks": 3, "score_threshold": 0.7},
)
```

Example with keyword search:
```python
response = await vector_io.query_chunks(
    vector_store_id="my_db",
    query="your query here",
    params={"mode": "keyword", "max_chunks": 3, "score_threshold": 0.7},
)
```

## Supported Search Modes

The SQLite vector store supports three search modes:

1. **Vector Search** (`mode="vector"`): Uses vector similarity to find relevant chunks
2. **Keyword Search** (`mode="keyword"`): Uses keyword matching to find relevant chunks
3. **Hybrid Search** (`mode="hybrid"`): Combines both vector and keyword scores using a ranker

### Hybrid Search

Hybrid search combines the strengths of both vector and keyword search by:
- Computing vector similarity scores
- Computing keyword match scores
- Using a ranker to combine these scores

Two ranker types are supported:

1. **RRF (Reciprocal Rank Fusion)**:
   - Combines ranks from both vector and keyword results
   - Uses an impact factor (default: 60.0) to control the weight of higher-ranked results
   - Good for balancing between vector and keyword results
   - The default impact factor of 60.0 comes from the original RRF paper by Cormack et al. (2009) [^1], which found this value to provide optimal performance across various retrieval tasks

2. **Weighted**:
   - Linearly combines normalized vector and keyword scores
   - Uses an alpha parameter (0-1) to control the blend:
     - alpha=0: Only use keyword scores
     - alpha=1: Only use vector scores
     - alpha=0.5: Equal weight to both (default)

Example using RAGQueryConfig with different search modes:

```python
from ogx_api import RAGQueryConfig, RRFRanker, WeightedRanker

# Vector search
config = RAGQueryConfig(mode="vector", max_chunks=5)

# Keyword search
config = RAGQueryConfig(mode="keyword", max_chunks=5)

# Hybrid search with custom RRF ranker
config = RAGQueryConfig(
    mode="hybrid",
    max_chunks=5,
    ranker=RRFRanker(impact_factor=50.0),  # Custom impact factor
)

# Hybrid search with weighted ranker
config = RAGQueryConfig(
    mode="hybrid",
    max_chunks=5,
    ranker=WeightedRanker(alpha=0.7),  # 70% vector, 30% keyword
)

# Hybrid search with default RRF ranker
config = RAGQueryConfig(
    mode="hybrid", max_chunks=5
)  # Will use RRF with impact_factor=60.0
```

Note: The ranker configuration is only used in hybrid mode. For vector or keyword modes, the ranker parameter is ignored.

## Installation

You can install SQLite-Vec using pip:

```bash
pip install sqlite-vec
```

## Documentation

See [sqlite-vec's GitHub repo](https://github.com/asg017/sqlite-vec/tree/main) for more details about sqlite-vec in general.

[^1]: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). [Reciprocal rank fusion outperforms condorcet and individual rank learning methods](https://dl.acm.org/doi/10.1145/1571941.1572114). In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (pp. 758-759).
""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="chromadb",
            provider_type="remote::chromadb",
            pip_packages=["chromadb-client"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.remote.vector_io.chroma",
            config_class="ogx.providers.remote.vector_io.chroma.ChromaVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
[Chroma](https://www.trychroma.com/) is an inline and remote vector
database provider for OGX. It allows you to store and query vectors directly within a Chroma database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features
Chroma supports:
- Store embeddings and their metadata
- Vector search
- Full-text search
- Document storage
- Metadata filtering
- Multi-modal retrieval

## Usage

To use Chrome in your OGX project, follow these steps:

1. Install the necessary dependencies.
2. Configure your OGX project to use chroma.
3. Start storing and querying vectors.

## Installation

You can install chroma using pip:

```bash
pip install chromadb
```

## Documentation
See [Chroma's documentation](https://docs.trychroma.com/docs/overview/introduction) for more details about Chroma in general.
""",
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::chromadb",
            pip_packages=["chromadb"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.inline.vector_io.chroma",
            config_class="ogx.providers.inline.vector_io.chroma.ChromaVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
[Chroma](https://www.trychroma.com/) is an inline and remote vector
database provider for OGX. It allows you to store and query vectors directly within a Chroma database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features
Chroma supports:
- Store embeddings and their metadata
- Vector search
- Full-text search
- Document storage
- Metadata filtering
- Multi-modal retrieval

## Usage

To use Chrome in your OGX project, follow these steps:

1. Install the necessary dependencies.
2. Configure your OGX project to use chroma.
3. Start storing and querying vectors.

## Installation

You can install chroma using pip:

```bash
pip install chromadb
```

## Documentation
See [Chroma's documentation](https://docs.trychroma.com/docs/overview/introduction) for more details about Chroma in general.

""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="pgvector",
            provider_type="remote::pgvector",
            pip_packages=["psycopg2-binary"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.remote.vector_io.pgvector",
            config_class="ogx.providers.remote.vector_io.pgvector.PGVectorVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
[PGVector](https://github.com/pgvector/pgvector) is a remote vector database provider for OGX. It
allows you to store and query vectors directly in memory.
That means you'll get fast and efficient vector retrieval.

## Features

- Easy to use
- Fully integrated with OGX

There are three implementations of search for PGVectoIndex available:

1. Vector Search:
- How it works:
  - Uses PostgreSQL's vector extension (pgvector) to perform similarity search
  - Compares query embeddings against stored embeddings using Cosine distance or other distance metrics
  - Eg. SQL query: SELECT document, embedding &lt;=&gt; %s::vector AS distance FROM table ORDER BY distance

-Characteristics:
  - Semantic understanding - finds documents similar in meaning even if they don't share keywords
  - Works with high-dimensional vector embeddings (typically 768, 1024, or higher dimensions)
  - Best for: Finding conceptually related content, handling synonyms, cross-language search
  - By default, OGX creates a HNSW (Hierarchical Navigable Small Worlds) index on a column "embedding" in a vector store table enabling production-ready, performant and scalable vector search for large datasets out of the box.

2. Keyword Search
- How it works:
  - Uses PostgreSQL's full-text search capabilities with tsvector and ts_rank
  - Converts text to searchable tokens using to_tsvector('english', text). Default language is English.
  - Eg. SQL query: SELECT document, ts_rank(tokenized_content, plainto_tsquery('english', %s)) AS score

- Characteristics:
  - Lexical matching - finds exact keyword matches and variations
  - Uses GIN (Generalized Inverted Index) for fast text search performance
  - Scoring: Uses PostgreSQL's ts_rank function for relevance scoring
  - Best for: Exact term matching, proper names, technical terms, Boolean-style queries

3. Hybrid Search
- How it works:
  - Combines both vector and keyword search results
  - Runs both searches independently, then merges results using configurable reranking

- Two reranking strategies available:
    - Reciprocal Rank Fusion (RRF) - (default: 60.0)
    - Weighted Average - (default: 0.5)

- Characteristics:
  - Best of both worlds: semantic understanding + exact matching
  - Documents appearing in both searches get boosted scores
  - Configurable balance between semantic and lexical matching
  - Best for: General-purpose search where you want both precision and recall

4. Database Schema

The PGVector implementation stores data optimized for all three search types:
CREATE TABLE vector_store_xxx (
    id TEXT PRIMARY KEY,
    document JSONB,                    -- Original document
    embedding vector(dimension),        -- For vector search
    content_text TEXT,                 -- Raw text content
    tokenized_content TSVECTOR          -- For keyword search
);


## Usage

To use PGVector in your OGX project, follow these steps:

1. Install the necessary dependencies.
2. Configure your OGX project to use pgvector. (e.g. remote::pgvector).
3. Start storing and querying vectors.

## This is an example how you can set up your environment for using PGVector (you can use either Podman or Docker)

1. Export PGVector environment variables:
```bash
export PGVECTOR_DB=testvectordb
export PGVECTOR_HOST=localhost
export PGVECTOR_PORT=5432
export PGVECTOR_USER=user
export PGVECTOR_PASSWORD=password
```

2. Pull pgvector image with that tag you want:

Via Podman:
```bash
podman pull pgvector/pgvector:0.8.1-pg18-trixie
```

Via Docker:
```bash
docker pull pgvector/pgvector:0.8.1-pg18-trixie
```

3. Run container with PGVector:

Via Podman
```bash
podman run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_USER=user \
  -e POSTGRES_DB=testvectordb \
  -p 5432:5432 \
  -v pgvector_data:/var/lib/postgresql \
  pgvector/pgvector:0.8.1-pg18-trixie
```

Via Docker
```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_USER=user \
  -e POSTGRES_DB=testvectordb \
  -p 5432:5432 \
  -v pgvector_data:/var/lib/postgresql \
  pgvector/pgvector:0.8.1-pg18-trixie
```

## Documentation
See [PGVector's documentation](https://github.com/pgvector/pgvector) for more details about PGVector in general.
""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="weaviate",
            provider_type="remote::weaviate",
            pip_packages=["weaviate-client>=4.16.5"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.remote.vector_io.weaviate",
            config_class="ogx.providers.remote.vector_io.weaviate.WeaviateVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
[Weaviate](https://weaviate.io/) is a vector database provider for OGX.
It allows you to store and query vectors directly within a Weaviate database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features
Weaviate supports:
- Store embeddings and their metadata
- Vector search
- Full-text search
- Hybrid search
- Document storage
- Metadata filtering
- Multi-modal retrieval


## Usage

To use Weaviate in your OGX project, follow these steps:

1. Install the necessary dependencies.
2. Configure your OGX project to use chroma.
3. Start storing and querying vectors.

## Installation

To install Weaviate see the [Weaviate quickstart documentation](https://weaviate.io/developers/weaviate/quickstart).

## Documentation
See [Weaviate's documentation](https://weaviate.io/developers/weaviate) for more details about Weaviate in general.
""",
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::qdrant",
            pip_packages=["qdrant-client"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.inline.vector_io.qdrant",
            config_class="ogx.providers.inline.vector_io.qdrant.QdrantVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description=r"""
[Qdrant](https://qdrant.tech/documentation/) is an inline and remote vector database provider for OGX. It
allows you to store and query vectors directly in memory.
That means you'll get fast and efficient vector retrieval.

> By default, Qdrant stores vectors in RAM, delivering incredibly fast access for datasets that fit comfortably in
> memory. But when your dataset exceeds RAM capacity, Qdrant offers Memmap as an alternative.
>
> \[[An Introduction to Vector Databases](https://qdrant.tech/articles/what-is-a-vector-database/)\]



## Features

- Lightweight and easy to use
- Fully integrated with OGX
- Apache 2.0 license terms
- Store embeddings and their metadata
- Supports search by
  [Keyword](https://qdrant.tech/articles/qdrant-introduces-full-text-filters-and-indexes/)
  and [Hybrid](https://qdrant.tech/articles/hybrid-search/#building-a-hybrid-search-system-in-qdrant) search
- [Multilingual and Multimodal retrieval](https://qdrant.tech/documentation/multimodal-search/)
- [Medatata filtering](https://qdrant.tech/articles/vector-search-filtering/)
- [GPU support](https://qdrant.tech/documentation/guides/running-with-gpu/)

## Usage

To use Qdrant in your OGX project, follow these steps:

1. Install the necessary dependencies.
2. Configure your OGX project to use Qdrant.
3. Start storing and querying vectors.

## Installation

You can install Qdrant using docker:

```bash
docker pull qdrant/qdrant
```
## Documentation
See the [Qdrant documentation](https://qdrant.tech/documentation/) for more details about Qdrant in general.
""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="qdrant",
            provider_type="remote::qdrant",
            pip_packages=["qdrant-client"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.remote.vector_io.qdrant",
            config_class="ogx.providers.remote.vector_io.qdrant.QdrantVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
Please refer to the inline provider documentation.
""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="milvus",
            provider_type="remote::milvus",
            pip_packages=["pymilvus>=2.6.2"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.remote.vector_io.milvus",
            config_class="ogx.providers.remote.vector_io.milvus.MilvusVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
[Milvus](https://milvus.io/) is an inline and remote vector database provider for OGX. It
allows you to store and query vectors directly within a Milvus database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features

- Easy to use
- Fully integrated with OGX
- Supports all search modes: vector, keyword, and hybrid search (both inline and remote configurations)

## Usage

To use Milvus in your OGX project, follow these steps:

1. Install the necessary dependencies.
2. Configure your OGX project to use Milvus.
3. Start storing and querying vectors.

## Installation

If you want to use inline Milvus, you can install:

```bash
pip install pymilvus[milvus-lite]
```

If you want to use remote Milvus, you can install:

```bash
pip install pymilvus
```

## Configuration

In OGX, Milvus can be configured in two ways:
- **Inline (Local) Configuration** - Uses Milvus-Lite for local storage
- **Remote Configuration** - Connects to a remote Milvus server

### Inline (Local) Configuration

The simplest method is local configuration, which requires setting `db_path`, a path for locally storing Milvus-Lite files:

```yaml
vector_io:
  - provider_id: milvus
    provider_type: inline::milvus
    config:
      db_path: ~/.llama/distributions/together/milvus_store.db
```

### Remote Configuration

Remote configuration is suitable for larger data storage requirements:

#### Standard Remote Connection

```yaml
vector_io:
  - provider_id: milvus
    provider_type: remote::milvus
    config:
      uri: "http://<host>:<port>"
      token: "<user>:<password>"
```

#### TLS-Enabled Remote Connection (One-way TLS)

For connections to Milvus instances with one-way TLS enabled:

```yaml
vector_io:
  - provider_id: milvus
    provider_type: remote::milvus
    config:
      uri: "https://<host>:<port>"
      token: "<user>:<password>"
      secure: True
      server_pem_path: "/path/to/server.pem"
```

#### Mutual TLS (mTLS) Remote Connection

For connections to Milvus instances with mutual TLS (mTLS) enabled:

```yaml
vector_io:
  - provider_id: milvus
    provider_type: remote::milvus
    config:
      uri: "https://<host>:<port>"
      token: "<user>:<password>"
      secure: True
      ca_pem_path: "/path/to/ca.pem"
      client_pem_path: "/path/to/client.pem"
      client_key_path: "/path/to/client.key"
```

#### Key Parameters for TLS Configuration

- **`secure`**: Enables TLS encryption when set to `true`. Defaults to `false`.
- **`server_pem_path`**: Path to the **server certificate** for verifying the server's identity (used in one-way TLS).
- **`ca_pem_path`**: Path to the **Certificate Authority (CA) certificate** for validating the server certificate (required in mTLS).
- **`client_pem_path`**: Path to the **client certificate** file (required for mTLS).
- **`client_key_path`**: Path to the **client private key** file (required for mTLS).

## Search Modes

Milvus supports three different search modes for both inline and remote configurations:

### Vector Search
Vector search uses semantic similarity to find the most relevant chunks based on embedding vectors. This is the default search mode and works well for finding conceptually similar content.

```python
# Vector search example
search_response = client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="What is machine learning?",
    search_mode="vector",
    max_num_results=5,
)
```

### Keyword Search
Keyword search uses traditional text-based matching to find chunks containing specific terms or phrases. This is useful when you need exact term matches.

```python
# Keyword search example
search_response = client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="Python programming language",
    search_mode="keyword",
    max_num_results=5,
)
```

### Hybrid Search
Hybrid search combines both vector and keyword search methods to provide more comprehensive results. It leverages the strengths of both semantic similarity and exact term matching.

#### Basic Hybrid Search
```python
# Basic hybrid search example (uses RRF ranker with default impact_factor=60.0)
search_response = client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="neural networks in Python",
    search_mode="hybrid",
    max_num_results=5,
)
```

**Note**: The default `impact_factor` value of 60.0 was empirically determined to be optimal in the original RRF research paper: ["Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (Cormack et al., 2009).

#### Hybrid Search with RRF (Reciprocal Rank Fusion) Ranker
RRF combines rankings from vector and keyword search by using reciprocal ranks. The impact factor controls how much weight is given to higher-ranked results.

```python
# Hybrid search with custom RRF parameters
search_response = client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="neural networks in Python",
    search_mode="hybrid",
    max_num_results=5,
    ranking_options={
        "ranker": {
            "type": "rrf",
            "impact_factor": 100.0,  # Higher values give more weight to top-ranked results
        }
    },
)
```

#### Hybrid Search with Weighted Ranker
Weighted ranker linearly combines normalized scores from vector and keyword search. The alpha parameter controls the balance between the two search methods.

```python
# Hybrid search with weighted ranker
search_response = client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="neural networks in Python",
    search_mode="hybrid",
    max_num_results=5,
    ranking_options={
        "ranker": {
            "type": "weighted",
            "alpha": 0.7,  # 70% vector search, 30% keyword search
        }
    },
)
```

For detailed documentation on RRF and Weighted rankers, please refer to the [Milvus Reranking Guide](https://milvus.io/docs/reranking.md).

## Documentation
See the [Milvus documentation](https://milvus.io/docs/install-overview.md) for more details about Milvus in general.

For more details on TLS configuration, refer to the [TLS setup guide](https://milvus.io/docs/tls.md).
""",
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::milvus",
            pip_packages=["pymilvus[milvus-lite]>=2.4.10"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.inline.vector_io.milvus",
            config_class="ogx.providers.inline.vector_io.milvus.MilvusVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
Please refer to the remote provider documentation.
""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="elasticsearch",
            provider_type="remote::elasticsearch",
            pip_packages=["elasticsearch>=8.16.0,<9.0.0"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.remote.vector_io.elasticsearch",
            config_class="ogx.providers.remote.vector_io.elasticsearch.ElasticsearchVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
[Elasticsearch](https://www.elastic.co/) is a vector database provider for OGX.
It allows you to store and query vectors directly within an Elasticsearch database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features
Elasticsearch supports:
- Store embeddings and their metadata
- Vector search
- Full-text search
- Fuzzy search
- Hybrid search
- Document storage
- Metadata filtering
- Inference service
- Machine Learning integrations

## Usage

To use Elasticsearch in your OGX project, follow these steps:

1. Install the necessary dependencies.
2. Configure your OGX project to use Elasticsearch.
3. Start storing and querying vectors.

## Installation

You can test Elasticsearch locally by running this script in the terminal:

```bash
curl -fsSL https://elastic.co/start-local | sh
```

Or you can [start a free trial](https://www.elastic.co/cloud/cloud-trial-overview?utm_campaign=ogx-integration) on Elastic Cloud.
For more information on how to deploy Elasticsearch, see the [official documentation](https://www.elastic.co/docs/deploy-manage/deploy).

## Documentation
See [Elasticsearch's documentation](https://www.elastic.co/docs/solutions/search) for more details about Elasticsearch in general.
""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="oci",
            provider_type="remote::oci",
            pip_packages=["oracledb", "numpy"] + DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.remote.vector_io.oci",
            config_class="ogx.providers.remote.vector_io.oci.OCI26aiVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
[Oracle 26ai](https://docs.oracle.com/en/database/oracle/oracle-database/26/index.html)
is a remote vector database provider for OGX. It allows you to store and query vectors directly
in an Oracle 26ai database.
## Features
- Easy to use
- Fully integrated with OGX
- Supports vector search, keyword search, and hybrid search
## Usage
To use Oracle 26ai in your OGX project, follow these steps:
1. Install the necessary dependencies.
2. Configure your OGX project to use Oracle 26ai.
3. Start storing and querying vectors.
## Installation
You can install the Oracle 26ai client using pip:
```bash
pip install oracledb
```
## Configuration
```yaml
vector_io:
- provider_id: oci
  provider_type: remote::oci
  config:
    conn_str: "${env.OCI26AI_CONNECTION_STRING}"
    user: "${env.OCI26AI_USER}"
    password: "${env.OCI26AI_PASSWORD}"
    tnsnames_loc: "${env.OCI26AI_TNSNAMES_LOC}"
    ewallet_pem_loc: "${env.OCI26AI_EWALLET_PEM_LOC}"
    ewallet_password: "${env.OCI26AI_EWALLET_PWD}"
    vector_datatype: "${env.OCI26AI_VECTOR_DATATYPE:=FLOAT32}"
    persistence:
      namespace: vector_id::oci26ai
      backend: kv_default
```
## Documentation
See the [Oracle 26ai documentation](https://docs.oracle.com/en/database/oracle/oracle-database/26/index.html)
for more details about Oracle 26ai in general.
""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="infinispan",
            provider_type="remote::infinispan",
            pip_packages=DEFAULT_VECTOR_IO_DEPS,
            module="ogx.providers.remote.vector_io.infinispan",
            config_class="ogx.providers.remote.vector_io.infinispan.InfinispanVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
            description="""
[Infinispan](https://infinispan.org/) is a remote vector database provider for OGX. It
allows you to store and query vectors in a distributed Infinispan cluster via HTTP REST API.
Infinispan provides high-performance, scalable data storage with support for both vector similarity
search and full-text search capabilities.

## Features

- **Vector Similarity Search** - Store and query embedding vectors with cosine similarity
- **Full-text/Keyword Search** - Query documents using Infinispan Query DSL or Ickle queries
- **Hybrid Search** - Combine vector and keyword search with configurable reranking (RRF or weighted)
- **Authentication** - Supports both Basic and Digest authentication mechanisms
- **HTTPS/TLS Support** - Secure connections with SSL certificate verification
- **Distributed Storage** - Leverage Infinispan's distributed caching for scalability
- **HTTP REST API** - Simple integration using standard HTTP protocol

## Search Modes

**Supported:**
- **Vector Search** (`mode="vector"`): Performs vector similarity search using embeddings
- **Keyword Search** (`mode="keyword"`): Full-text search using Infinispan Query DSL/Ickle
- **Hybrid Search** (`mode="hybrid"`): Combines vector and keyword search with configurable reranking

## Configuration

### Basic Configuration (HTTP)

```yaml
vector_io:
  - provider_id: infinispan
    provider_type: remote::infinispan
    config:
      url: "http://localhost:11222"
      username: "admin"
      password: "password"
      auth_mechanism: "digest"
      persistence:
        backend: "kv_default"
        namespace: "vector_io::infinispan"
```

### HTTPS Configuration with TLS

```yaml
vector_io:
  - provider_id: infinispan
    provider_type: remote::infinispan
    config:
      url: "https://infinispan.example.com:11222"
      username: "admin"
      password: "password"
      use_https: true
      auth_mechanism: "basic"
      verify_tls: true
      persistence:
        backend: "kv_default"
        namespace: "vector_io::infinispan"
```

### Environment Variables

You can use environment variables for sensitive configuration:

```yaml
vector_io:
  - provider_id: infinispan
    provider_type: remote::infinispan
    config:
      url: "${env.INFINISPAN_URL}"
      username: "${env.INFINISPAN_USERNAME}"
      password: "${env.INFINISPAN_PASSWORD}"
      persistence:
        backend: "kv_default"
        namespace: "vector_io::infinispan"
```

## Usage

### Starting Infinispan Server

The easiest way to get started is using Docker:

```bash
docker run -it -p 11222:11222 \\
  -e USER="admin" \\
  -e PASS="password" \\
  infinispan/server:latest
```

## Authentication

Infinispan supports two authentication mechanisms:

- **Digest Authentication** (recommended for HTTP): More secure than basic auth over HTTP
- **Basic Authentication** (HTTPS only): Simple username/password authentication

Set the `auth_mechanism` parameter to either `"digest"` or `"basic"`.

## Documentation

- [Infinispan Documentation](https://infinispan.org/documentation/)
- [Infinispan REST API Guide](https://infinispan.org/docs/stable/titles/rest/rest.html)
- [Infinispan Vector Search](https://infinispan.org/docs/stable/titles/developing/developing.html#vector-search)
- [Infinispan Query DSL](https://infinispan.org/docs/stable/titles/query/query.html)

## Requirements

- Infinispan Server 16.0+ (with vector search support)
""",
        ),
    ]
