# Containerfile used to build our all in one ollama image to run tests in CI
#
# podman build --platform linux/amd64 -f ./ollama-with-models.containerfile -t ollama-with-models .
#
FROM --platform=linux/amd64 ollama/ollama:0.34.2

# Start ollama and pull models in a single layer
RUN ollama serve & \
    sleep 5 && \
    ollama pull llama3.2:3b-instruct-fp16 && \
    ollama pull nomic-embed-text:v1.5 && \
    ollama pull llama-guard3:1b

# Set the entrypoint to start ollama serve
ENTRYPOINT ["ollama", "serve"]
