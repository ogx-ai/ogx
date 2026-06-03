# OpenAPI Generator SDK

Alternative SDK generation using [OpenAPI Generator](https://github.com/OpenAPITools/openapi-generator) instead of Stainless. See [#4609](https://github.com/ogx-ai/ogx/issues/4609) for context.

## Prerequisites

### Java 11+

openapi-generator-cli requires Java 11 or higher.

```bash
# macOS
brew install openjdk

# Fedora/RHEL/CentOS
sudo dnf install java-11-openjdk
# For other Linux distributions, use your package manager (apt, yum, pacman, etc.).
```

### OpenAPI Generator CLI

For more installation options, see: <https://openapi-generator.tech/docs/installation>

```bash
# macOS
brew install openapi-generator

# Linux (also possible for macOS)
npm install -g @openapitools/openapi-generator-cli
```

### Python Dependencies

```bash
uv pip install ruamel.yaml
```

## Usage

**From the client-sdks/openapi directory:**

```bash
cd client-sdks/openapi

make openapi    # Generate enriched OpenAPI spec from Stainless config
make hierarchy  # Process spec for hierarchical SDK structure
make sdk        # Generate Python SDK (runs full pipeline)
make version    # Show version that will be used
make clean      # Remove generated files
```

The `make sdk` target runs the full pipeline and will automatically check for required dependencies (openapi-generator-cli and java) before generating.

## How it Works

```text
merge_stainless_config.py  ->  build_hierarchy.py  ->  openapi-generator  ->  patch_hierarchy.py
```

1. **`merge_stainless_config.py`** reads base spec from `../stainless/openapi.yml`, enriches it with resource mappings from `../stainless/config.yml`, and applies patches from `patches.yml`. This is the only step that depends on the Stainless config.
2. **`build_hierarchy.py`** extracts tag hierarchies, reduces endpoints to leaf tags, creates dummy endpoints for parent resource groups, and applies schema fixes for openapi-generator compatibility.
3. **`openapi-generator`** generates the Python SDK from the processed spec using custom Mustache templates.
4. **`patch_hierarchy.py`** patches the generated API classes to wire up parent-child relationships, enabling nested access like `client.chat.completions.create(...)`.

**Generated files (git-ignored):**

- `openapi.yml` - Enriched OpenAPI specification
- `openapi-hierarchical.yml` - Processed spec with hierarchy tags
- `api-hierarchy.yml` - Hierarchy data for post-generation patching
- `sdks/python/` - Generated Python SDK
- `.openapi-generator/` - Generator metadata

## CI/CD Automation

### Continuous Integration

The CI workflow (`.github/workflows/openapi-generator-validation.yml`) automatically validates SDK generation on every PR:

- ✅ Generates OpenAPI spec from Stainless config
- ✅ Builds Python SDK (1,134 files)
- ✅ Verifies SDK installation and imports
- ✅ Runs integration tests against generated SDK
- ✅ Multi-platform testing:
  - **Ubuntu** - Always runs
  - **macOS** - Runs for main/release branch pushes, or when critical files change

**Triggered by:**

- Pull requests modifying OpenAPI generation files
- Pushes to `main` or `release-*` branches
- Manual workflow_dispatch

### Continuous Delivery

The CD workflow (`.github/workflows/publish-openapi-sdk.yml`) automatically publishes SDK to PyPI:

**Automatic publishing (via tags):**

- Tags matching `openapi-sdk-v*` trigger builds
- Stable versions (e.g., `openapi-sdk-v1.0.0`) → Published to TestPyPI
- Pre-release versions (e.g., `openapi-sdk-v1.0.0-rc1`) → Built only, not published

**Manual publishing (via GitHub UI):**

- Go to Actions → "Publish OpenAPI SDK to PyPI"
- Choose target (TestPyPI/PyPI) and dry-run mode

## Publishing to PyPI

The SDK can be published to PyPI using the GitHub Actions workflow at `.github/workflows/publish-openapi-sdk.yml`.

### Manual Publishing (via GitHub UI)

1. Go to Actions → "Publish OpenAPI SDK to PyPI"
2. Click "Run workflow"
3. Select options:
   - **publish_to**: `testpypi` (for testing) or `pypi` (production)
   - **dry_run**: `true` to build only without publishing

### Automatic Publishing (via Git Tags)

Push a tag matching `openapi-sdk-v*` to trigger automatic builds:

```bash
# Stable release → Published to TestPyPI
git tag openapi-sdk-v1.0.0
git push origin openapi-sdk-v1.0.0

# Pre-release → Built only, not published
git tag openapi-sdk-v1.0.0-rc1
git push origin openapi-sdk-v1.0.0-rc1
```

**Note:** Pre-release tags (containing `-rc`, `-alpha`, or `-beta`) are built for validation but not published to avoid cluttering the package index.

### Required Secrets

Configure these GitHub secrets for the repository:

- `TEST_PYPI_API_TOKEN` - TestPyPI API token
- `PYPI_API_TOKEN` - Production PyPI API token

### Testing the Published Package

After publishing to TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ ogx-open-client
```

After publishing to PyPI:

```bash
pip install ogx-open-client
```

## Documentation

- **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - End-to-end code examples for all major API features
- **[STRATEGY.md](STRATEGY.md)** - Long-term strategy, ownership, versioning, and deprecation policy
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment guide, environment setup, rollback procedures

## Files

- `Makefile` - Build orchestration
- `merge_stainless_config.py` - Merge Stainless config into OpenAPI spec
- `build_hierarchy.py` - Extract hierarchy and prepare spec for code generation
- `patch_hierarchy.py` - Post-generation patching for nested API structure
- `patches.yml` - OpenAPI spec patches for codegen compatibility
- `openapi-config.json` - Python SDK generation config
- `openapitools.json` - OpenAPI Generator CLI version config
- `templates/python/` - Custom Mustache templates and library files
  - `LICENSE.mustache` - MIT license for generated SDK
  - `CHANGELOG.mustache` - Changelog template for release notes
