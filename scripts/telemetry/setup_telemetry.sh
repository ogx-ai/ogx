#!/usr/bin/env bash

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Telemetry Setup Script for OGX
# This script sets up Jaeger, OpenTelemetry Collector, Prometheus, and Grafana using Podman
# For whoever is interested in testing the telemetry stack, you can run this script to set up the stack.
#    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
#    export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
#    export OTEL_SERVICE_NAME=my-llama-app
# Then run the distro server

set -Eeuo pipefail

# Parse arguments
CONTAINER_RUNTIME=""

print_usage() {
  echo "Usage: $0 [--container docker|podman]"
  echo ""
  echo "Options:"
  echo "  -c, --container    Choose container runtime (docker or podman)."
  echo "  -h, --help         Show this help."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--container)
      if [[ $# -lt 2 ]]; then
        echo "🚨 --container requires a value: docker or podman"
        exit 1
      fi
      case "$2" in
        docker|podman)
          CONTAINER_RUNTIME="$2"
          shift 2
          ;;
        *)
          echo "🚨 Invalid container runtime: $2"
          echo "Valid options are: docker, podman"
          exit 1
          ;;
      esac
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "🚨 Unknown argument: $1"
      print_usage
      exit 1
      ;;
  esac
done

# Detect container runtime if not specified
if [[ -z "$CONTAINER_RUNTIME" ]]; then
  if command -v podman &> /dev/null; then
    CONTAINER_RUNTIME="podman"
  elif command -v docker &> /dev/null; then
    CONTAINER_RUNTIME="docker"
  else
    echo "🚨 Neither Podman nor Docker could be found"
    echo "Install Docker: https://docs.docker.com/get-docker/ or Podman: https://podman.io/getting-started/installation"
    exit 1
  fi
fi

echo "🚀 Setting up telemetry stack for OGX using $CONTAINER_RUNTIME..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v "$CONTAINER_RUNTIME" &> /dev/null; then
  echo "🚨 $CONTAINER_RUNTIME could not be found"
  echo "Docker or Podman is required. Install Docker: https://docs.docker.com/get-docker/ or Podman: https://podman.io/getting-started/installation"
  exit 1
fi

# Create a network for the services
echo "📡 Creating $CONTAINER_RUNTIME network..."
$CONTAINER_RUNTIME network create llama-telemetry 2>/dev/null || echo "Network already exists"

# Stop and remove existing containers
echo "🧹 Cleaning up existing containers..."
$CONTAINER_RUNTIME stop jaeger otel-collector prometheus grafana mlflow 2>/dev/null || true
$CONTAINER_RUNTIME rm jaeger otel-collector prometheus grafana mlflow 2>/dev/null || true

# Temp paths for MLflow storage (auto-created under /tmp)
STATE_DIR="${STATE_DIR:-/tmp/llama-telemetry}"
MLFLOW_BACKEND_STORE="$STATE_DIR/mlflow/mlflow.db"
MLFLOW_ARTIFACT_ROOT="$STATE_DIR/mlruns"

# Ensure paths exist for bind mounts
mkdir -p "$(dirname "$MLFLOW_BACKEND_STORE")"
touch "$MLFLOW_BACKEND_STORE"
mkdir -p "$MLFLOW_ARTIFACT_ROOT"

# Experiment ID for MLflow traces (collector header)
MLFLOW_EXPERIMENT_ID_STR="${MLFLOW_EXPERIMENT_ID_STR:-\"0\"}"
MLFLOW_OTEL_HEADERS="${MLFLOW_OTEL_HEADERS:-}"

# Start Jaeger
echo "🔍 Starting Jaeger..."
$CONTAINER_RUNTIME run -d --name jaeger \
  --network llama-telemetry \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 16686:16686 \
  -p 14250:14250 \
  -p 9411:9411 \
  docker.io/jaegertracing/all-in-one:latest

# Start MLflow server (containerized)
echo "📒 Starting MLflow..."
$CONTAINER_RUNTIME run -d --name mlflow \
  --network llama-telemetry \
  -p 5001:5001 \
  -v "$MLFLOW_BACKEND_STORE:/mlflow/mlflow.db" \
  -v "$MLFLOW_ARTIFACT_ROOT:/mlflow/artifacts" \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 --port 5001 \
    --allowed-hosts localhost,localhost:5001,127.0.0.1,127.0.0.1:5001,host.docker.internal,host.docker.internal:5001

# Add host aliases so the Collector can reach host services (e.g., MLflow)
ADD_HOST_OPT=""
if [[ "$CONTAINER_RUNTIME" == "docker" ]]; then
  ADD_HOST_OPT="--add-host=host.docker.internal:host-gateway"
elif [[ "$CONTAINER_RUNTIME" == "podman" ]]; then
  ADD_HOST_OPT="--add-host=host.containers.internal:host-gateway"
fi

# Start OpenTelemetry Collector
echo "📊 Starting OpenTelemetry Collector..."
$CONTAINER_RUNTIME run -d --name otel-collector \
  --network llama-telemetry \
  $ADD_HOST_OPT \
  -p 4318:4318 \
  -p 4317:4317 \
  -p 9464:9464 \
  -p 13133:13133 \
  -e "MLFLOW_EXPERIMENT_ID_STR=$MLFLOW_EXPERIMENT_ID_STR" \
  -e "MLFLOW_OTEL_HEADERS=$MLFLOW_OTEL_HEADERS" \
  -v "$SCRIPT_DIR/otel-collector-config.yaml:/etc/otel-collector-config.yaml:Z" \
  docker.io/otel/opentelemetry-collector-contrib:latest \
  --config /etc/otel-collector-config.yaml

# Start Prometheus
echo "📈 Starting Prometheus..."
$CONTAINER_RUNTIME run -d --name prometheus \
  --network llama-telemetry \
  -p 9090:9090 \
  -v "$SCRIPT_DIR/prometheus.yml:/etc/prometheus/prometheus.yml:Z" \
  docker.io/prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.console.libraries=/etc/prometheus/console_libraries \
  --web.console.templates=/etc/prometheus/consoles \
  --storage.tsdb.retention.time=200h \
  --web.enable-lifecycle

# Start Grafana
# Note: Using 11.0.0 because grafana:latest arm64 image has a broken /run.sh (0 bytes)
echo "📊 Starting Grafana..."
$CONTAINER_RUNTIME run -d --name grafana \
  --network llama-telemetry \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  -e GF_USERS_ALLOW_SIGN_UP=false \
  -v "$SCRIPT_DIR/grafana-datasources.yaml:/etc/grafana/provisioning/datasources/datasources.yaml:Z" \
  -v "$SCRIPT_DIR/grafana-dashboards.yaml:/etc/grafana/provisioning/dashboards/dashboards.yaml:Z" \
  -v "$SCRIPT_DIR/ogx-dashboard.json:/etc/grafana/provisioning/dashboards/ogx-dashboard.json:Z" \
  -v "$SCRIPT_DIR/ogx-tool-runtime-metrics.json:/etc/grafana/provisioning/dashboards/ogx-tool-runtime-metrics.json:Z" \
  -v "$SCRIPT_DIR/ogx-vector-io-metrics.json:/etc/grafana/provisioning/dashboards/ogx-vector-io-metrics.json:Z" \
  -v "$SCRIPT_DIR/ogx-request-metrics.json:/etc/grafana/provisioning/dashboards/ogx-request-metrics.json:Z" \
  -v "$SCRIPT_DIR/ogx-responses-metrics.json:/etc/grafana/provisioning/dashboards/ogx-responses-metrics.json:Z" \
  -v "$SCRIPT_DIR/ogx-inference-metrics.json:/etc/grafana/provisioning/dashboards/ogx-inference-metrics.json:Z" \
  docker.io/grafana/grafana:11.0.0

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service status..."
$CONTAINER_RUNTIME ps --filter "name=jaeger|otel-collector|prometheus|grafana|mlflow" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "✅ Telemetry stack is ready!"
echo ""
echo "🌐 Service URLs:"
echo "   MLflow:           http://localhost:5001"
echo "   Jaeger UI:        http://localhost:16686"
echo "   Prometheus:       http://localhost:9090"
echo "   Grafana:          http://localhost:3000 (admin/admin)"
echo "   OTEL Collector:   http://localhost:4318 (OTLP endpoint)"
echo ""
echo "🔧 Environment variables for OGX:"
echo "   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318"
echo "   export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf"
echo "   export OTEL_SERVICE_NAME=my-llama-app"
echo ""
echo "📊 Next steps:"
echo "   1. Set the environment variables above"
echo "   2. Start your OGX application"
echo "   3. Make some inference calls to generate metrics"
echo "   4. Check Jaeger for traces: http://localhost:16686"
echo "   5. Check Prometheus for metrics: http://localhost:9090"
echo "   6. Set up Grafana dashboards: http://localhost:3000"
echo ""
echo "🔍 To test the setup, run:"
echo "   curl -X POST http://localhost:5001/v1/inference/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model_id\": \"your-model\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
echo "🧹 To clean up when done:"
echo "   $CONTAINER_RUNTIME stop jaeger otel-collector prometheus grafana mlflow"
echo "   $CONTAINER_RUNTIME rm jaeger otel-collector prometheus grafana mlflow"
echo "   $CONTAINER_RUNTIME network rm llama-telemetry"
echo "   rm -rf $STATE_DIR  # remove temp DB/artifacts"
