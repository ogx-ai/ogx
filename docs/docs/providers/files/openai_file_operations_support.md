# File Operations Support in Vector Store Providers

## Overview

This document provides a comprehensive overview of file operations and Vector Store API support across all available vector store providers in OGX. As of release 0.2.24, the following providers support full file operations integration.

## Supported Providers

### ✅ Full File Operations Support

The following providers support complete file operations integration, including file upload, automatic processing, and search:

#### Inline Providers (Single Node)

| Provider | File Operations | Key Features |
|----------|----------------|--------------|
| **FAISS** | ✅ Full Support | Fast in-memory search, GPU acceleration |
| **SQLite-vec** | ✅ Full Support | Hybrid search, disk-based storage |
| **Milvus** | ✅ Full Support | High-performance, scalable indexing |

#### Remote Providers (Hosted)

| Provider | File Operations | Key Features |
|----------|----------------|--------------|
| **ChromaDB** | ✅ Full Support | Metadata filtering, persistent storage |
| **Qdrant** | ✅ Full Support | Payload filtering, advanced search |
| **Weaviate** | ✅ Full Support | GraphQL interface, schema management |
| **Postgres (PGVector)** | ✅ Full Support | SQL integration, ACID compliance |

### 🔄 Partial Support

Some providers may support basic vector operations but lack full file operations integration:

| Provider | Status | Notes |
|----------|--------|-------|
| **Builtin** | 🔄 Basic | Core vector operations only |

## File Operations Features

All supported providers offer the following file operations capabilities:

### Core Functionality

- **File Upload & Processing**: Automatic document ingestion and chunking
- **Vector Storage**: Embedding generation and storage
- **Search & Retrieval**: Semantic search with metadata filtering
- **File Management**: List, retrieve, and manage files in vector stores

### Advanced Features

- **Automatic Chunking**: Configurable chunk sizes and overlap
- **Metadata Preservation**: File attributes and chunk metadata
- **Status Tracking**: Monitor file processing progress
- **Error Handling**: Comprehensive error reporting and recovery

## Implementation Details

### File Processing Pipeline

1. **Upload**: File uploaded via Files API
2. **Extraction**: Text content extracted from various formats
3. **Chunking**: Content split into optimal chunks (default: 800 tokens)
4. **Embedding**: Chunks converted to vector embeddings
5. **Storage**: Vectors stored with metadata in vector database
6. **Indexing**: Search index updated for fast retrieval

### Supported File Formats

- **Documents**: PDF, DOCX, DOC
- **Text**: TXT, MD, RST
- **Code**: Python, JavaScript, Java, C++, etc.
- **Data**: JSON, CSV, XML
- **Web**: HTML files

### Chunking Strategies

- **Default**: 800 tokens with 400 token overlap
- **Custom**: Configurable chunk sizes and overlap
- **Static**: Fixed-size chunks with overlap

## Provider-Specific Features

### FAISS

- **Storage**: In-memory with optional persistence
- **Performance**: Optimized for speed and GPU acceleration
- **Use Case**: High-performance, memory-constrained environments

### SQLite-vec

- **Storage**: Disk-based with SQLite backend
- **Search**: Hybrid vector + keyword search
- **Use Case**: Large document collections, frequent updates

### Milvus

- **Storage**: Scalable distributed storage
- **Indexing**: Multiple index types (IVF, HNSW)
- **Use Case**: Production deployments, large-scale applications

### ChromaDB

- **Storage**: Persistent storage with metadata
- **Filtering**: Advanced metadata filtering
- **Use Case**: Applications requiring rich metadata

### Qdrant

- **Storage**: High-performance vector database
- **Filtering**: Payload-based filtering
- **Use Case**: Real-time applications, complex queries

### Weaviate

- **Storage**: GraphQL-native vector database
- **Schema**: Flexible schema management
- **Use Case**: Applications requiring complex data relationships

### Postgres (PGVector)

- **Storage**: SQL database with vector extensions
- **Integration**: ACID compliance, existing SQL workflows
- **Use Case**: Applications requiring transactional guarantees

## Configuration Examples

### Basic Configuration

```yaml
vector_io:
  - provider_id: faiss
    provider_type: inline::faiss
    config:
      kvstore:
        type: sqlite
        db_path: ~/.llama/faiss_store.db
```

### With FileResponse Support

```yaml
vector_io:
  - provider_id: faiss
    provider_type: inline::faiss
    config:
      kvstore:
        type: sqlite
        db_path: ~/.llama/faiss_store.db

files:
  - provider_id: local-files
    provider_type: inline::localfs
    config:
      storage_dir: ~/.llama/files
      metadata_store:
        type: sqlite
        db_path: ~/.llama/files_metadata.db
```

## Usage Examples

### Python Client

```python
from ogx import LlamaStackClient

client = LlamaStackClient("http://localhost:8000")

# Create vector store
vector_store = client.vector_stores.create(name="documents")

# Upload and process file
with open("document.pdf", "rb") as f:
    file_info = await client.files.upload(file=f, purpose="assistants")

# Attach to vector store
await client.vector_stores.files.create(
    vector_store_id=vector_store.id, file_id=file_info.id
)

# Search
results = await client.vector_stores.search(
    vector_store_id=vector_store.id, query="What is the main topic?", max_num_results=5
)
```

### cURL Commands

```bash
# Upload file
curl -X POST http://localhost:8000/v1/openai/v1/files \
  -F "file=@document.pdf" \
  -F "purpose=assistants"

# Create vector store
curl -X POST http://localhost:8000/v1/openai/v1/vector_stores \
  -H "Content-Type: application/json" \
  -d '{"name": "documents"}'

# Attach file to vector store
curl -X POST http://localhost:8000/v1/openai/v1/vector_stores/{store_id}/files \
  -H "Content-Type: application/json" \
  -d '{"file_id": "file-abc123"}'

# Search vector store
curl -X POST http://localhost:8000/v1/openai/v1/vector_stores/{store_id}/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "max_num_results": 5}'
```

## Performance Considerations

### Chunk Size Optimization

- **Small chunks (400-600 tokens)**: Better precision, more results
- **Large chunks (800-1200 tokens)**: Better context, fewer results
- **Overlap (50%)**: Maintains context between chunks

### Storage Efficiency

- **FAISS**: Fastest, but memory-limited
- **SQLite-vec**: Good balance of performance and storage
- **Milvus**: Scalable, production-ready
- **Remote providers**: Managed, but network-dependent

### Search Performance

- **Vector search**: Fastest for semantic queries
- **Hybrid search**: Best accuracy (SQLite-vec only)
- **Filtered search**: Fast with metadata constraints

## Troubleshooting

### Common Issues

1. **File Processing Failures**
   - Check file format compatibility
   - Verify file size limits
   - Review error messages in file status

2. **Search Performance**
   - Optimize chunk sizes for your use case
   - Use filters to narrow search scope
   - Monitor vector store metrics

3. **Storage Issues**
   - Check available disk space
   - Verify database permissions
   - Monitor memory usage (for in-memory providers)

### Monitoring

```python
# Check file processing status
file_status = await client.vector_stores.files.retrieve(
    vector_store_id=vector_store.id, file_id=file_info.id
)

if file_status.status == "failed":
    print(f"Error: {file_status.last_error.message}")

# Monitor vector store health
health = await client.vector_stores.health(vector_store_id=vector_store.id)
print(f"Status: {health.status}")
```

## Best Practices

1. **File Organization**: Use descriptive names and organize by purpose
2. **Chunking Strategy**: Test different sizes for your specific use case
3. **Metadata**: Add relevant attributes for better filtering
4. **Monitoring**: Track processing status and search performance
5. **Cleanup**: Regularly remove unused files to manage storage

## Future Enhancements

Planned improvements for file operations support:

- **Batch Processing**: Process multiple files simultaneously
- **Advanced Chunking**: More sophisticated chunking algorithms
- **Custom Embeddings**: Support for custom embedding models
- **Real-time Updates**: Live file processing and indexing
- **Multi-format Support**: Enhanced file format support

## Support and Resources

- **Documentation**: [File Operations and Vector Store Integration](../../concepts/file_operations_vector_stores)
- **API Reference**: Files API
- **Provider Docs**: [Vector Store Providers](../vector_io/)
- **Examples**: [Getting Started](../../getting_started/)
- **Community**: [GitHub Issues](https://github.com/ogx-ai/ogx/issues)
