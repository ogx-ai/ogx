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

## Files

- `Makefile` - Build orchestration
- `merge_stainless_config.py` - Merge Stainless config into OpenAPI spec
- `build_hierarchy.py` - Extract hierarchy and prepare spec for code generation
- `patch_hierarchy.py` - Post-generation patching for nested API structure
- `patches.yml` - OpenAPI spec patches for codegen compatibility
- `openapi-config.json` - Python SDK generation config
- `openapitools.json` - OpenAPI Generator CLI version config
- `templates/python/` - Custom Mustache templates and library files
