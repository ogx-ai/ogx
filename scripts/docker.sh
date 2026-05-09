#!/usr/bin/env bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Docker container management script for OGX
# Allows starting/stopping/restarting a OGX docker container for testing

# Default values
DISTRO=""
PORT=8321
INFERENCE_MODE="replay"
COMMAND=""
USE_COPY_NOT_MOUNT=false
NO_REBUILD=false
PLATFORM=""

# Function to display usage
usage() {
    cat <<EOF
Usage: $0 COMMAND [OPTIONS]

Commands:
    start       Build and start the docker container
    stop        Stop and remove the docker container
    restart     Restart the docker container
    status      Check if the container is running
    logs        Show container logs (add -f to follow)

Options:
    --distro STRING          Distribution name (e.g., 'ci-tests', 'starter') (required for start/restart)
    --port NUMBER            Port to run on (default: 8321)
    --inference-mode STRING  Inference mode: replay, record-if-missing or record (default: replay)
    --copy-source            Copy source into image instead of mounting (default: auto-detect CI, otherwise mount)
    --no-rebuild             Skip building the image, just start the container (default: false)
    --platform STRING        Target platform for build (e.g., 'linux/arm64', 'linux/amd64'). Uses docker buildx or podman buildx if specified.
    --help                   Show this help message

Examples:
    # Start a docker container (local dev mode - mounts source, builds image)
    $0 start --distro ci-tests

    # Start without rebuilding (uses existing image)
    $0 start --distro ci-tests --no-rebuild

    # Start with source copied into image (like CI)
    $0 start --distro ci-tests --copy-source

    # Start with custom port
    $0 start --distro starter --port 8080

    # Build for ARM64 platforms
    $0 start --distro ci-tests --platform linux/arm64

    # Check status
    $0 status --distro ci-tests

    # View logs
    $0 logs --distro ci-tests

    # Stop container
    $0 stop --distro ci-tests

    # Restart container
    $0 restart --distro ci-tests

Note: In CI environments (detected via CI or GITHUB_ACTIONS env vars), source is
      automatically copied into the image. Locally, source is mounted for live development
      unless --copy-source is specified.
EOF
}

# Parse command (first positional arg)
if [[ $# -eq 0 ]]; then
    echo "Error: Command required"
    usage
    exit 1
fi

COMMAND="$1"
shift

# Validate command
case "$COMMAND" in
start | stop | restart | status | logs) ;;
--help)
    usage
    exit 0
    ;;
*)
    echo "Error: Unknown command: $COMMAND"
    usage
    exit 1
    ;;
esac

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
    --distro)
        DISTRO="$2"
        shift 2
        ;;
    --port)
        PORT="$2"
        shift 2
        ;;
    --inference-mode)
        INFERENCE_MODE="$2"
        shift 2
        ;;
    --copy-source)
        USE_COPY_NOT_MOUNT=true
        shift
        ;;
    --no-rebuild)
        NO_REBUILD=true
        shift
        ;;
    --platform)
        PLATFORM="$2"
        shift 2
        ;;
    --help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
done

# Validate required parameters for commands that need them
if [[ "$COMMAND" != "stop" && "$COMMAND" != "status" && "$COMMAND" != "logs" ]]; then
    if [[ -z "$DISTRO" ]]; then
        echo "Error: --distro is required for '$COMMAND' command"
        usage
        exit 1
    fi
fi

# If distro not provided for stop/status/logs, try to infer from running containers
if [[ -z "$DISTRO" && ("$COMMAND" == "stop" || "$COMMAND" == "status" || "$COMMAND" == "logs") ]]; then
    # Look for any ogx-test-* container
    RUNNING_CONTAINERS=$(docker ps -a --filter "name=ogx-test-" --format "{{.Names}}" | head -1)
    if [[ -n "$RUNNING_CONTAINERS" ]]; then
        DISTRO=$(echo "$RUNNING_CONTAINERS" | sed 's/ogx-test-//')
        echo "Found running container for distro: $DISTRO"
    else
        echo "Error: --distro is required (no running containers found)"
        usage
        exit 1
    fi
fi

# Remove docker: prefix if present
DISTRO=$(echo "$DISTRO" | sed 's/^docker://')

CONTAINER_NAME="ogx-test-$DISTRO"

should_copy_source() {
    if [[ "$USE_COPY_NOT_MOUNT" == "true" ]]; then
        return 0
    fi
    if [[ "${CI:-false}" == "true" ]] || [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
        return 0
    fi
    return 1
}

# Function to check if container is running
is_container_running() {
    docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Function to check if container exists (running or stopped)
container_exists() {
    docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Function to stop and remove container
stop_container() {
    if container_exists; then
        echo "Stopping container: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        echo "Removing container: $CONTAINER_NAME"
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
        echo "✅ Container stopped and removed"
    else
        echo "⚠️  Container $CONTAINER_NAME does not exist"
    fi
}

# Function to build docker image
build_image() {
    echo "=== Building Docker Image for distribution: $DISTRO ==="
    # Get the repo root (parent of scripts directory)
    local script_dir
    script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    local repo_root
    repo_root=$(cd "$script_dir/.." && pwd)

    local containerfile="$repo_root/containers/Containerfile"
    if [[ ! -f "$containerfile" ]]; then
        echo "❌ Containerfile not found at $containerfile"
        exit 1
    fi

    # Determine version for labels
    local version
    if ! version=$(git describe --tags --always --dirty 2>&1); then
        echo "❌ Failed to determine version from git: $version"
        exit 1
    fi

    # Generate config labels (one per line, read into array)
    echo "Generating config labels for distribution: $DISTRO"
    local config_labels_output
    if ! config_labels_output=$("$script_dir/generate-config-labels.sh" "$DISTRO" "$version"); then
        echo "❌ Failed to generate config labels"
        exit 1
    fi
    local config_labels
    mapfile -t config_labels <<< "$config_labels_output"

    # Determine if we should use buildx for multi-arch builds
    local use_buildx=false
    if [[ -n "$PLATFORM" ]]; then
        use_buildx=true
        echo "Building for platform: $PLATFORM (using docker buildx)"
        # Ensure buildx is available
        if ! docker buildx version &>/dev/null; then
            echo "Error: docker buildx or podman buildx is required for --platform builds"
            echo "Install Docker Buildx (or update Docker to a version that includes it) or Podman with buildx support"
            exit 1
        fi
        # Create buildx builder if it doesn't exist
        docker buildx create --name ogx-builder --use 2>/dev/null || docker buildx use ogx-builder 2>/dev/null || true
    fi

    local build_cmd
    if [[ "$use_buildx" == "true" ]]; then
        build_cmd=(
            docker
            buildx
            build
            --platform "$PLATFORM"
            --load
            "$repo_root"
            -f "$containerfile"
            --tag "localhost/distribution-$DISTRO:dev"
            --build-arg "DISTRO_NAME=$DISTRO"
            --build-arg "INSTALL_MODE=editable"
            --build-arg "OGX_DIR=/workspace"
        )
    else
        build_cmd=(
            docker
            build
            "$repo_root"
            -f "$containerfile"
            --tag "localhost/distribution-$DISTRO:dev"
            --build-arg "DISTRO_NAME=$DISTRO"
            --build-arg "INSTALL_MODE=editable"
            --build-arg "OGX_DIR=/workspace"
        )
    fi

    # Pass UV index configuration for release branches
    if [[ -n "${UV_EXTRA_INDEX_URL:-}" ]]; then
        echo "Adding UV_EXTRA_INDEX_URL to docker build: $UV_EXTRA_INDEX_URL"
        build_cmd+=(--build-arg "UV_EXTRA_INDEX_URL=$UV_EXTRA_INDEX_URL")
    fi
    if [[ -n "${UV_INDEX_STRATEGY:-}" ]]; then
        echo "Adding UV_INDEX_STRATEGY to docker build: $UV_INDEX_STRATEGY"
        build_cmd+=(--build-arg "UV_INDEX_STRATEGY=$UV_INDEX_STRATEGY")
    fi

    # Add config labels to build command
    build_cmd+=("${config_labels[@]}")

    if ! "${build_cmd[@]}"; then
        echo "❌ Failed to build Docker image"
        exit 1
    fi
    echo "✅ Docker image built successfully with embedded config labels"
}

# Function to start container
start_container() {
    # Check if already running
    if is_container_running; then
        echo "⚠️  Container $CONTAINER_NAME is already running"
        echo "URL: http://localhost:$PORT"
        exit 0
    fi

    # Stop and remove if exists but not running
    if container_exists; then
        echo "Removing existing stopped container..."
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi

    # Build the image unless --no-rebuild was specified
    if [[ "$NO_REBUILD" == "true" ]]; then
        echo "Skipping build (--no-rebuild specified)"
        # Check if image exists (with or without localhost/ prefix)
        if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "distribution-$DISTRO:dev$"; then
            echo "❌ Error: Image distribution-$DISTRO:dev does not exist"
            echo "Either build it first without --no-rebuild, or run: docker build . -f containers/Containerfile --build-arg DISTRO_NAME=$DISTRO --tag localhost/distribution-$DISTRO:dev"
            exit 1
        fi
        echo "✅ Found existing image for distribution-$DISTRO:dev"
    else
        build_image
    fi

    echo ""
    echo "=== Starting Docker Container ==="

    # Get the repo root for volume mount
    local script_dir
    script_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
    local repo_root
    repo_root=$(cd "$script_dir/.." && pwd)

    # Determine the actual image name (may have localhost/ prefix)
    IMAGE_NAME=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "distribution-$DISTRO:dev$" | head -1)
    if [[ -z "$IMAGE_NAME" ]]; then
        echo "❌ Error: Could not find image for distribution-$DISTRO:dev"
        exit 1
    fi
    echo "Using image: $IMAGE_NAME"

    # Build environment variables for docker run
    DOCKER_ENV_VARS=""
    DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e OGX_TEST_INFERENCE_MODE=$INFERENCE_MODE"
    DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e OGX_TEST_STACK_CONFIG_TYPE=server"

    # Set default OLLAMA_URL if not provided
    # On macOS/Windows, use host.docker.internal to reach host from container
    # On Linux with --network host, use localhost
    if [[ "$(uname)" == "Darwin" ]] || [[ "$(uname)" == *"MINGW"* ]]; then
        OLLAMA_URL="${OLLAMA_URL:-http://host.docker.internal:11434/v1}"
    else
        OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/v1}"
    fi
    DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e OLLAMA_URL=$OLLAMA_URL"

    # Pass through API keys if they exist
    [ -n "${TOGETHER_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e TOGETHER_API_KEY=$TOGETHER_API_KEY"
    [ -n "${FIREWORKS_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY"
    [ -n "${TAVILY_SEARCH_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e TAVILY_SEARCH_API_KEY=$TAVILY_SEARCH_API_KEY"
    [ -n "${OPENAI_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e OPENAI_API_KEY=$OPENAI_API_KEY"
    [ -n "${ANTHROPIC_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
    [ -n "${GROQ_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e GROQ_API_KEY=$GROQ_API_KEY"
    [ -n "${GEMINI_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e GEMINI_API_KEY=$GEMINI_API_KEY"
    [ -n "${SAFETY_MODEL:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e SAFETY_MODEL=$SAFETY_MODEL"
    [ -n "${SQLITE_STORE_DIR:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e SQLITE_STORE_DIR=$SQLITE_STORE_DIR"

    # Use --network host on Linux only (macOS doesn't support it properly)
    NETWORK_MODE=""
    if [[ "$(uname)" != "Darwin" ]] && [[ "$(uname)" != *"MINGW"* ]]; then
        NETWORK_MODE="--network host"
    fi

    local source_mount=""
    if should_copy_source; then
        echo "Source baked into image (no volume mount)"
    else
        source_mount="-v \"$repo_root\":/workspace"
        echo "Mounting $repo_root into /workspace"
    fi

    docker run -d $NETWORK_MODE --name "$CONTAINER_NAME" \
        -p $PORT:$PORT \
        $DOCKER_ENV_VARS \
        $source_mount \
        "$IMAGE_NAME" \
        --port $PORT

    echo "Waiting for container to start..."
    for i in {1..30}; do
        if curl -s http://localhost:$PORT/v1/health 2>/dev/null | grep -q "OK"; then
            echo "✅ Container started successfully"
            echo ""
            echo "=== Container Information ==="
            echo "Container name: $CONTAINER_NAME"
            echo "URL: http://localhost:$PORT"
            echo "Health check: http://localhost:$PORT/v1/health"
            echo ""
            echo "To view logs: $0 logs --distro $DISTRO"
            echo "To stop: $0 stop --distro $DISTRO"
            return 0
        fi
        if [[ $i -eq 30 ]]; then
            echo "❌ Container failed to start within timeout"
            echo "Showing container logs:"
            docker logs "$CONTAINER_NAME"
            exit 1
        fi
        sleep 1
    done
}

# Execute command
case "$COMMAND" in
start)
    start_container
    ;;
stop)
    stop_container
    ;;
restart)
    echo "Restarting container: $CONTAINER_NAME"
    stop_container
    echo ""
    start_container
    ;;
status)
    if is_container_running; then
        echo "✅ Container $CONTAINER_NAME is running"
        echo "URL: http://localhost:$PORT"
        # Try to get the actual port from the container
        ACTUAL_PORT=$(docker port "$CONTAINER_NAME" 2>/dev/null | grep "8321/tcp" | cut -d':' -f2 | head -1)
        if [[ -n "$ACTUAL_PORT" ]]; then
            echo "Port: $ACTUAL_PORT"
        fi
    elif container_exists; then
        echo "⚠️  Container $CONTAINER_NAME exists but is not running"
        echo "Start it with: $0 start --distro $DISTRO"
    else
        echo "❌ Container $CONTAINER_NAME does not exist"
        echo "Start it with: $0 start --distro $DISTRO"
    fi
    ;;
logs)
    if container_exists; then
        echo "=== Logs for $CONTAINER_NAME ==="
        # Check if -f flag was passed after 'logs' command
        if [[ "${1:-}" == "-f" || "${1:-}" == "--follow" ]]; then
            docker logs --tail 100 --follow "$CONTAINER_NAME"
        else
            docker logs --tail 100 "$CONTAINER_NAME"
        fi
    else
        echo "❌ Container $CONTAINER_NAME does not exist"
        exit 1
    fi
    ;;
esac
