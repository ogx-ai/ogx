# distributions

Pre-built distribution configurations that wire together specific providers for common deployment scenarios.

## Directory Structure

```text
distributions/
  starter/             # General-purpose distribution with many providers enabled
  ci-tests/            # Minimal distribution for CI testing
  nvidia/              # NVIDIA NIM-based distribution
  oci/                 # Oracle Cloud Infrastructure distribution
  open-benchmark/      # Benchmarking distribution
  postgres-demo/       # PostgreSQL-backed distribution demo
  watsonx/             # IBM WatsonX distribution
  __init__.py
  template.py          # Distribution template rendering engine
```

## What Is a Distribution

A distribution is a pre-built configuration that bundles specific providers for a target environment, similar to how Kubernetes has distributions like AKS, EKS, and GKE. The core API stays the same, but each distribution wires different backends. Concretely, it is a `config.yaml` file that defines which APIs to serve, which providers to use for each API, how storage is configured, and what models/shields/datasets to register at startup.

Example from `starter/config.yaml`:

```yaml
version: 2
distro_name: starter
apis: [inference, agents, safety, vector_io, ...]
providers:
  inference:
    - provider_id: ollama
      provider_type: remote::ollama
      config:
        base_url: ${env.OLLAMA_URL:=http://localhost:11434/v1}
```

## Environment Variable Substitution

Distribution configs use `${env.VAR:=default}` syntax for environment-driven configuration. The `${env.VAR:+value}` syntax conditionally enables providers only when the variable is set.

## Template System

`template.py` provides the `DistributionTemplate` class used to generate distribution configs programmatically. Each distribution directory can contain a template definition that `scripts/distro_codegen.py` uses to regenerate `config.yaml` files.

## Usage

Run a distribution with:

```bash
ogx run starter
# or
ogx run --config path/to/config.yaml
```
