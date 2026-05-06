# File Operations Quick Reference

## Overview

As of release 0.2.14, OGX provides comprehensive file operations and Vector Store API integration, following the [OpenAI Vector Store Files API specification](https://platform.openai.com/docs/api-reference/vector-stores-files).

> **Note**: For detailed overview and implementation details, see [Overview](openai_file_operations_support.md#overview) in the full documentation.

## Supported Providers

> **Note**: For complete provider details and features, see [Supported Providers](openai_file_operations_support.md#supported-providers) in the full documentation.

**Inline Providers**: FAISS, SQLite-vec, Milvus
**Remote Providers**: ChromaDB, Qdrant, Weaviate, PGVector

## Quick Start

### 1. Upload File

```python
file_info = await client.files.upload(
    file=open("document.pdf", "rb"), purpose="assistants"
)
```

### 2. Create Vector Store

```python
vector_store = client.vector_stores.create(name="my_docs")
```

### 3. Attach File

```python
await client.vector_stores.files.create(
    vector_store_id=vector_store.id, file_id=file_info.id
)
```

### 4. Search

```python
results = await client.vector_stores.search(
    vector_store_id=vector_store.id, query="What is the main topic?", max_num_results=5
)
```

## File Processing & Search

**Processing**: 800 tokens default chunk size, 400 token overlap
**Formats**: PDF, DOCX, TXT, Code files, etc.
**Search**: Vector similarity, Hybrid (SQLite-vec), Filtered with metadata

## Configuration

> **Note**: For detailed configuration examples and options, see [Configuration Examples](openai_file_operations_support.md#configuration-examples) in the full documentation.

**Basic Setup**: Configure vector_io and files providers in your config.yaml

## Common Use Cases

- **RAG Systems**: Document Q&A with file uploads
- **Knowledge Bases**: Searchable document collections
- **Content Analysis**: Document similarity and clustering
- **Research Tools**: Literature review and analysis

## Performance Tips

> **Note**: For detailed performance optimization strategies, see [Performance Considerations](openai_file_operations_support.md#performance-considerations) in the full documentation.

**Quick Tips**: Choose provider based on your needs (speed vs. storage vs. scalability)

## Troubleshooting

> **Note**: For comprehensive troubleshooting, see [Troubleshooting](openai_file_operations_support.md#troubleshooting) in the full documentation.

**Quick Fixes**: Check file format compatibility, optimize chunk sizes, monitor storage

## Resources

- [Full Documentation](openai_file_operations_support.md)
- [Integration Guide](../../concepts/file_operations_vector_stores)
- Files API
- [Provider Details](../vector_io/)
