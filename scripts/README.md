# scripts

Build, codegen, and CI utility scripts.

## Directory Structure

```text
scripts/
  openapi_generator/           # OpenAPI schema generation tools
  telemetry/                   # Telemetry dashboard setup scripts
  distro_codegen.py            # Regenerate distribution config.yaml files from templates
  provider_codegen.py          # Regenerate provider registry and routing code
  generate_ci_matrix.py        # Generate CI test matrix from provider/distribution data
  gen-ci-docs.py               # Generate CI documentation
  generate_target_models_docs.py # Generate integration target model documentation
  run_openapi_generator.sh     # Run OpenAPI spec generation
  check-api-conformance.sh     # Check API conformance
  check_file_size.py           # Check file sizes
  check-init-py.sh             # Verify all packages have __init__.py
  check-no-fstring-logging.sh  # Verify no f-string logging usage
  check-workflows-use-hashes.sh # Verify GitHub Actions use commit hashes
  cleanup_recordings.py        # Remove orphaned test recordings (incl. stale model-list recordings)
  diagnose_recordings.py       # Debug test recording issues
  normalize_recordings.py      # Normalize test recordings for consistency
  docker.sh                    # Docker build helper
  install.sh                   # Installation helper
  integration-tests.sh         # Run integration test suite
  quickstart_smoke.py          # Run every Quickstart combination (docs steps.json) against a fresh install
  unit-tests.sh                # Run unit test suite
  run-ui-linter.sh             # Run UI linter
  uv-run-with-index.sh         # Run uv with custom package index
  get_setup_env.py             # Get setup environment variables
  generate_prompt_format.py    # Generate prompt format documentation
  generate_openai_coverage_docs.py # Generate OpenAI API coverage docs
   openai_coverage.py           # OpenAI API coverage analysis
  refresh_openresponses_spec.py # Regenerate docs/static/openresponses-spec.json from OpenAI's spec
   systemone_coverage.py        # TypeSafe SystemOne API conformance analysis
   provider_compat_matrix.py    # Provider compatibility matrix
  responses_test_coverage.py   # Responses API test coverage
```

## Common Operations

### Regenerate distribution configs

```bash
uv run python scripts/distro_codegen.py
```

### Regenerate provider registry

```bash
uv run python scripts/provider_codegen.py
```

### Refresh the OpenAI Responses coverage reference

```bash
curl -sL -o /tmp/openai-openapi.yaml \
    https://raw.githubusercontent.com/openai/openai-openapi/main/openapi.yaml
uv run python scripts/refresh_openresponses_spec.py /tmp/openai-openapi.yaml
uv run python scripts/openai_coverage.py --update --generate-docs
```

### Run tests

```bash
./scripts/unit-tests.sh
./scripts/integration-tests.sh --stack-config starter
```

Pass `--client-version latest` to generate the Python client from
`client-sdks/openapi` and install it over the PyPI pin before running
(the CI `client-version=latest` equivalent), or `--client-version published`
to verify the `uv.lock`-resolved PyPI client is installed.

### Run the docs Quickstart

The Quickstart page renders its steps from
`docs/docs/getting_started/quickstart/steps.json`, and `quickstart_smoke.py` runs the
same steps for every supported combination of backend, server mode, and use case. To
change the Quickstart, edit `steps.json` and the Python files next to it, then run:

```bash
# Every combination, and which ones the page marks unsupported
uv run --no-project python scripts/quickstart_smoke.py --list

# The Ollama combinations, from wheels built from this checkout
uv build --wheel --out-dir dist && uv build --wheel --out-dir dist src/ogx_api
uv run --no-project python scripts/quickstart_smoke.py --wheels dist --backend ollama
```

Without `--wheels`, the server comes from the latest PyPI release. The OpenAI,
Anthropic, and "All of them" combinations need `OPENAI_API_KEY` and/or
`ANTHROPIC_API_KEY` and are skipped without them. Coding-agent combinations need
the agent CLIs on `PATH`, and Ollama needs `OLLAMA_CONTEXT_LENGTH=16384` or more.
Set `QUICKSTART_OLLAMA_CMD` when Ollama runs in a container (CI uses
`docker exec ollama ollama`).

### Manage test recordings

```bash
# Clean up orphaned recordings
uv run python scripts/cleanup_recordings.py
# Diagnose recording issues
uv run python scripts/diagnose_recordings.py
# Normalize recordings
uv run python scripts/normalize_recordings.py

# Only check model-list recordings (skips CI test collection)
uv run python scripts/cleanup_recordings.py --model-lists-only
```

`cleanup_recordings.py` also prunes model-list recordings (`models-*.json`):
files whose filename hash no longer matches the current hash function are dead
(no replay lookup can find them), and model-set digests that add nothing the
kept files of the same (directory, request hash) do not cover are redundant,
since replay unions all digests of one hash. Only model IDs referenced by the
test suites (recording requests + CI stack-config registrations) are required
to survive; every kept group still covers all of them.

### Remote test recording (via GitHub Actions)

```bash
./scripts/github/schedule-record-workflow.sh --test-subdirs "inference,agents"
```
