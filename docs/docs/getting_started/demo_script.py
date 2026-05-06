# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_client import Agent, AgentEventLogger, LlamaStackClient

vector_store_id = "my_demo_vector_db"
client = LlamaStackClient(base_url="http://localhost:8321")

models = client.models.list()

# Select the first LLM and first embedding models
# Prefer Ollama models since they don't require API keys
models_list = list(models)
llm_models = [m for m in models_list if m.id and not m.id.startswith("sentence-transformers")]
ollama_models = [m for m in llm_models if "ollama" in m.id.lower()]
model_id = (ollama_models[0] if ollama_models else llm_models[0]).id

# Get embedding model
embedding_models = [m for m in models_list if m.id and m.id.startswith("sentence-transformers")]
em = embedding_models[0] if embedding_models else None
if not em:
    raise ValueError("No embedding model found")
embedding_model_id = em.id
# Default embedding dimension for nomic-embed-text-v1.5 is 768
embedding_dimension = 768

print(f"Using model: {model_id}")

# Download the document content
import requests
source_url = "https://www.paulgraham.com/greatwork.html"
print(f"Downloading document: {source_url}")
response = requests.get(source_url)
content = response.text

# Upload the file
print("Uploading file to server...")
file_obj = client.files.create(
    file=("greatwork.html", content.encode('utf-8'), "text/html"),
    purpose="assistants",
)
file_id = file_obj.id
print(f"File uploaded: {file_id}")

# Create or retrieve vector store
print(f"Creating vector store: {vector_store_id}")
try:
    # Try to retrieve existing vector store
    vector_store = client.vector_stores.retrieve(vector_store_id)
    print(f"Using existing vector store: {vector_store_id}")
    vector_store_id = vector_store.id

    # Add file to existing vector store
    client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_id,
    )
    print(f"Added file to vector store")
except Exception as e:
    # Create new vector store with the file
    print(f"Creating new vector store (error: {e})")
    vector_store = client.vector_stores.create(
        name=vector_store_id,
        file_ids=[file_id],
    )
    vector_store_id = vector_store.id
    print(f"Created new vector store: {vector_store_id}")
agent = Agent(
    client,
    model=model_id,
    instructions="You are a helpful assistant. Use the file_search tool to find relevant information in the ingested documents.",
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
        }
    ],
)

prompt = "How do you do great work?"
print("prompt>", prompt)

use_stream = True
response = agent.create_turn(
    messages=[{"role": "user", "content": prompt}],
    session_id=agent.create_session("rag_session"),
    stream=use_stream,
)

# Only call `AgentEventLogger().log(response)` for streaming responses.
if use_stream:
    for log in AgentEventLogger().log(response):
        if hasattr(log, 'print'):
            log.print()
        else:
            # Print text chunks inline without newlines
            print(log, end='', flush=True)
    print()  # Final newline at the end
else:
    print(response)
