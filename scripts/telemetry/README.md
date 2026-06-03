# Observability Test for OGX

This directory contains configuration files and a setup script to deploy a full observability stack for testing OGX telemetry. It uses OpenTelemetry, Jaeger, Prometheus, and Grafana to collect traces, metrics, and visualize them.

## Architecture

```text
┌──────────────┐
│  OGX │──┐
│  Server      │  │  OTLP     ┌───────────────────┐   scrape    ┌──────────────┐
└──────────────┘  ├─────────►│  OTel Collector     │◄───────────│  Prometheus  │
┌──────────────┐  │  :4318   │  :4317 (gRPC)       │   :9464    │  :9090       │
│  OGX │──┘          │  :4318 (HTTP)       │            └──────────────┘
│  Client      │             └────────┬────────────┘                   ▲
└──────────────┘                      │ OTLP                           │
                                      ▼                           datasource
                             ┌────────────────┐                        │
                             │  Jaeger         │              ┌──────────────┐
                             │  :16686 (UI)    │◄────────────│  Grafana     │
                             └────────────────┘  datasource   │  :3000 (UI)  │
                                                              └──────────────┘
```

| Component | Purpose | Port(s) |
|---|---|---|
| **OTel Collector** | Receives OTLP telemetry, exports traces to Jaeger and metrics to Prometheus (and optionally MLflow) | 4317 (gRPC), 4318 (HTTP), 9464 (Prometheus metrics) |
| **Jaeger** | Distributed tracing UI | 16686 |
| **Prometheus** | Metrics storage and querying | 9090 |
| **Grafana** | Dashboards and visualization | 3000 |
| **MLflow** | Trace ingest via OTLP `/v1/traces` (container in this stack) | 5001 |

## Pre-requisites

- **Docker** or **Podman** installed and running
- **OGX** development environment set up
- **OpenTelemetry Python packages** (installed in step 2 below)

## Files

| File | Description |
|---|---|
| `setup_telemetry.sh` | Main script to start all telemetry services |
| `otel-collector-config.yaml` | OpenTelemetry Collector configuration (OTLP receiver, Prometheus exporter, Jaeger exporter) |
| `prometheus.yml` | Prometheus scrape configuration (scrapes OTel Collector on port 9464) |
| `grafana-datasources.yaml` | Grafana datasource provisioning (Prometheus + Jaeger) |
| `grafana-dashboards.yaml` | Grafana dashboard provisioning configuration |
| `ogx-dashboard.json` | Pre-built Grafana dashboard for OGX (token usage, operation duration, HTTP metrics) |

## Steps

### Start the telemetry stack

Run the setup script to start Jaeger, OTel Collector, Prometheus, and Grafana:

```bash
# Auto-detect container runtime (podman or docker)
./scripts/telemetry/setup_telemetry.sh

# Or specify a container runtime explicitly
./scripts/telemetry/setup_telemetry.sh --container docker
./scripts/telemetry/setup_telemetry.sh --container podman
```

This will:

- Create a `llama-telemetry` container network
- Start Jaeger, OTel Collector, Prometheus, and Grafana containers
- Provision Grafana with a pre-built OGX dashboard

> **MLflow traces**
>
> - MLflow is now started as a container in this stack (`mlflow:5001`), OTLP endpoint `/v1/traces`.
> - Collector exporter `otlphttp/mlflow` points to `http://mlflow:5001/v1/traces`, header `x-mlflow-experiment-id: "1"`. If you need auth, set `MLFLOW_OTEL_HEADERS` (e.g., `Authorization=Bearer <token>`) before running the setup script.
> - If you prefer an external MLflow, override `MLFLOW_OTEL_ENDPOINT` before running the script (e.g., `http://host.docker.internal:5001`).

### Install OpenTelemetry instrumentation For OGX Server and Client

```bash
uv pip install opentelemetry-distro opentelemetry-exporter-otlp
uv run opentelemetry-bootstrap -a requirements | uv pip install --requirement -
```

### Set environment variables and start OGX Server

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_SERVICE_NAME=ogx-server

uv run opentelemetry-instrument ogx stack run starter
```

> **Note:** The `opentelemetry-instrument` wrapper automatically instruments the application and sends traces/metrics to the OTel Collector.

### Set environment variables and start OGX Client

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_SERVICE_NAME=my-ogx-app

opentelemetry-instrument python ogx-client.py
```

An example `ogx-client.py`

```python
from openai import OpenAI


def main():
    client = OpenAI(
        api_key="fake",
        base_url="http://localhost:8321/v1/",
    )
    print("hello")
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )
    print("Sync response: ", response.choices[0].message.content)


if __name__ == "__main__":
    main()
```

### Explore the telemetry data

Open the following UIs in your browser:

| Service | URL | Credentials |
|---|---|---|
| **Mlflow** (traces) | [http://localhost:5001](http://localhost:5001) | N/A |
| **Jaeger** (traces) | [http://localhost:16686](http://localhost:16686) | N/A |
| **Prometheus** (metrics) | [http://localhost:9090](http://localhost:9090) | N/A |
| **Grafana** (dashboards) | [http://localhost:3000](http://localhost:3000) | admin / admin |

#### Prometheus metrics to try

The following metrics are automatically generated by OpenTelemetry auto-instrumentation:

```promql
# Total input token usage by model
sum by(gen_ai_request_model) (ogx_gen_ai_client_token_usage_sum{gen_ai_token_type="input"})

# Total output token usage by model
sum by(gen_ai_request_model) (ogx_gen_ai_client_token_usage_sum{gen_ai_token_type="output"})

# P95 HTTP server latency
histogram_quantile(0.95, rate(ogx_http_server_duration_milliseconds_bucket[5m]))

# Total HTTP request count
sum(ogx_http_server_duration_milliseconds_count)
```

#### Grafana dashboard

A pre-provisioned **OGX** dashboard is available in Grafana with panels for:

- Prompt Tokens (input token usage by model)
- Completion Tokens (output token usage by model)
- P95 / P99 HTTP Server Duration
- Total HTTP Requests

### GenAI message content capture (logs vs spans)

- `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` controls whether prompts/outputs/tool-call args are captured. **Default: false** (no content captured). Set to `true` to capture.
- Captured content is emitted as **logs** (e.g., events like `gen_ai.user.message` and `gen_ai.choice`), with `trace_id`/`span_id` for correlation.
- Spans carry structured metadata (model, finish_reason, usage tokens, latency, HTTP call, etc.) but **not** the raw text content.

### Exporter examples

**Local console (debugging):**

```bash
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true \
OTEL_TRACES_EXPORTER=console \
OTEL_LOGS_EXPORTER=console \
opentelemetry-instrument python ogx-client.py
```

**Send to Collector (recommended):**

```bash
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true \
OTEL_TRACES_EXPORTER=otlp \
OTEL_LOGS_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
opentelemetry-instrument python ogx-client.py
```

### Jaeger caveat (logs)

- Jaeger’s built-in OTLP typically ingests **traces only**, not logs. If you point `OTEL_LOGS_EXPORTER=otlp` at Jaeger, logs will be rejected (e.g., 404) and you will not see message content.
- To view captured content, send logs to an OTel Collector (or another backend that supports OTLP logs). For debugging, you can also use `OTEL_LOGS_EXPORTER=console` to print logs to stdout.

### Minimal Collector example (traces → Jaeger, logs → stdout)

```yaml
receivers:
  otlp:
    protocols:
      http:
      grpc:

processors:
  batch: {}

exporters:
  jaeger:
    endpoint: localhost:14250
    tls:
      insecure: true
  logging:  # view logs in collector stdout
    loglevel: info

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [logging]
```

Use with:

```bash
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true \
OTEL_TRACES_EXPORTER=otlp \
OTEL_LOGS_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
opentelemetry-instrument python ogx-client.py
```

## Cleanup

Stop and remove all telemetry containers:

```bash
# Replace "docker" with "podman" if applicable
docker stop jaeger otel-collector prometheus grafana
docker rm jaeger otel-collector prometheus grafana
docker network rm llama-telemetry
```

## Known Issues

When OpenTelemetry auto-instrumentation is enabled, both the low-level database driver instrumentor
(e.g. `asyncpg`, `sqlite3`) and the SQLAlchemy ORM instrumentor activate simultaneously. This causes
duplicate spans and inflated traces. To prevent this, disable the driver-level instrumentors:

```bash
export OTEL_PYTHON_DISABLED_INSTRUMENTATIONS="sqlite3,asyncpg"
```

## References

- [OpenTelemetry Documentation](https://opentelemetry.io/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [Jaeger Documentation](https://www.jaegertracing.io/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
