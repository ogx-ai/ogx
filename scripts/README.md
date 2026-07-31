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
  cleanup_recordings.py        # Remove orphaned test recordings
  diagnose_recordings.py       # Debug test recording issues
  normalize_recordings.py      # Normalize test recordings for consistency
  docker.sh                    # Docker build helper
  install.sh                   # Installation helper
  integration-tests.sh         # Run integration test suite
  unit-tests.sh                # Run unit test suite
  run-ui-linter.sh             # Run UI linter
  uv-run-with-index.sh         # Run uv with custom package index
  get_setup_env.py             # Get setup environment variables
  generate_prompt_format.py    # Generate prompt format documentation
  generate_openai_coverage_docs.py # Generate OpenAI API coverage docs
  openai_coverage.py           # OpenAI API coverage analysis
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

### Run tests

```bash
./scripts/unit-tests.sh
./scripts/integration-tests.sh --stack-config starter
```

### Manage test recordings

```bash
# Clean up orphaned recordings
uv run python scripts/cleanup_recordings.py
# Diagnose recording issues
uv run python scripts/diagnose_recordings.py
# Normalize recordings
uv run python scripts/normalize_recordings.py
```

### Remote test recording (via GitHub Actions)

```bash
./scripts/github/schedule-record-workflow.sh --test-subdirs "inference,agents"
```
