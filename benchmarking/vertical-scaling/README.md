# OGX Vertical Scaling Benchmark

A simple benchmark suite for measuring the vertical scaling performance of the OGX inference API using a mocked OpenAI-compatible backend.

## Overview

This benchmark measures how well the OGX server scales with increasing worker counts. It uses:

- **Python Mock Server** - Lightweight HTTP server returning fixed OpenAI-compatible responses
- **Locust** - Load testing tool to generate concurrent requests
- **OGX Server** - The inference API under test with `remote::openai` provider

The benchmark runs twice:

1. **Baseline**: Tests the mock server directly to establish maximum throughput
2. **Stack**: Tests the stack server to measure routing overhead

This helps identify bottlenecks in the stack routing layer and measure the impact of vertical scaling (adding more workers) on throughput and latency.

## Prerequisites

### Python (for mock server)

The mock server is a simple Python script with no dependencies beyond the standard library.

```bash
python --version
```

### Locust (Load Testing)

Verify installation:

```bash
uv run --group benchmark locust --version
```

## Quick Start

Run a benchmark with default settings (1 worker, 1 user, 60 seconds):

```bash
cd benchmarking/vertical-scaling
./run-benchmark.sh
```

This will:

1. Check port availability (8080, 8321)
2. Start Python mock server on port 8080
3. Start OGX server on port 8321 with 1 worker
4. Run baseline benchmark against mock server (60 seconds)
5. Run stack benchmark against stack server (60 seconds)
6. Generate HTML reports in `results/`
7. Clean up all services

## Usage

**Server Options:**

- `--workers NUMBER` - Number of uvicorn workers (default: 1)
- `--mock-port NUMBER` - Port for mock server (default: 8080)
- `--stack-port NUMBER` - Port for OGX server (default: 8321)

**Benchmark Options:**

- `--users NUMBER` - Concurrent users (default: 1)
- `--run-time SECONDS` - Test duration (default: 60)

## Interpreting Results

### Output Files

Results are saved to `results/` directory:

- `baseline.html` - Interactive HTML report for mock server baseline
- `baseline_stats.csv` - Baseline summary statistics
- `baseline_stats_history.csv` - Baseline time-series data
- `baseline_failures.csv` - Baseline failed requests
- `stack.html` - Interactive HTML report for stack server
- `stack_stats.csv` - Stack summary statistics
- `stack_stats_history.csv` - Stack time-series data
- `stack_failures.csv` - Stack failed requests

Additional files in the benchmark directory:

- `stack.log` - OGX server logs

Compare the baseline and stack HTML reports to see the routing overhead.

### Key Metrics

Open the HTML report to view:

1. **Requests/s (RPS)** - Throughput metric
   - Higher is better
   - Should increase with more workers (up to a point)

2. **Response Times** - Latency metrics
   - **p50 (median)** - Typical user experience
   - **p95** - 95th percentile, catches slower requests
   - **p99** - 99th percentile, catches outliers
   - Lower is better

3. **Failure Rate** - Error percentage
   - Should be 0% for healthy setup
   - High failure rate indicates overload

### Identifying Bottlenecks

**Good vertical scaling:**

- RPS increases linearly with workers (2x workers ≈ 2x RPS)
- Latency remains stable or improves
- No failures

**CPU-bound bottleneck:**

- RPS increases but plateaus at high worker counts
- Latency increases with more workers
- System CPU usage at 100%

**I/O-bound bottleneck:**

- RPS doesn't increase much with workers
- Latency increases significantly
- Network or disk I/O saturated

**GIL bottleneck (Python):**

- RPS plateaus around 2-4 workers
- Adding more workers doesn't help
- Consider using multiple processes
