# OpenAPI Generator SDK

The `ogx-client` Python SDK is generated with [OpenAPI Generator](https://github.com/OpenAPITools/openapi-generator). The base OpenAPI spec and resource-hierarchy config are read from `../spec` and enriched before code generation.

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

make openapi    # Generate enriched OpenAPI spec from the base spec + resource config
make hierarchy  # Process spec for hierarchical SDK structure
make sdk        # Generate Python SDK (runs full pipeline)
make version    # Show version that will be used
make clean      # Remove generated files
```

The `make sdk` target runs the full pipeline and will automatically check for required dependencies (openapi-generator-cli and java) before generating.

## How it Works

```text
merge_resources.py  ->  build_hierarchy.py  ->  openapi-generator  ->  patch_hierarchy.py
```

1. **`merge_resources.py`** reads the base spec from `../spec/openapi.yml`, enriches it with resource mappings from `../spec/resources.yml`, and applies patches from `patches.yml`. This is the only step that depends on the resource-hierarchy config.
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

- ✅ Generates OpenAPI spec from the base spec + resource config
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

The `ogx-client` package is published through the unified PyPI/NPM release workflow (`.github/workflows/pypi.yml`) alongside other ogx packages.

**Automatic publishing (via tags):**

- Tags matching `v*` trigger the unified workflow
- The workflow builds all packages including `ogx-client`

**Manual publishing (via GitHub UI):**

- Go to Actions → "Build, test, and publish packages"
- Choose `packages: clients-only` (or `all`) and the desired `dry_run` mode

## Publishing to PyPI

The SDK is published as `ogx-client` via the unified workflow at `.github/workflows/pypi.yml`.

### Manual Publishing (via GitHub UI)

1. Go to Actions → "Build, test, and publish packages"
2. Click "Run workflow"
3. Select options:
   - **packages**: `clients-only` or `all`
   - **dry_run**: `test-pypi` (default), `build-only`, or `off` (production)

### Automatic Publishing (via Git Tags)

Push a version tag to trigger the unified workflow:

```bash
# Release → triggers unified workflow for all packages
git tag v1.0.0
git push origin v1.0.0
```

### Testing the Published Package

After publishing to TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ ogx-client
```

After publishing to PyPI:

```bash
pip install ogx-client
```

## Files

- `Makefile` - Build orchestration
- `merge_resources.py` - Merge the resource-hierarchy config into the OpenAPI spec
- `build_hierarchy.py` - Extract hierarchy and prepare spec for code generation
- `patch_hierarchy.py` - Post-generation patching for nested API structure
- `patches.yml` - OpenAPI spec patches for codegen compatibility
- `openapi-config.json` - Python SDK generation config
- `openapitools.json` - OpenAPI Generator CLI version config
- `templates/python/` - Custom Mustache templates and library files
  - `LICENSE.mustache` - MIT license for generated SDK
  - `CHANGELOG.mustache` - Changelog template for release notes
