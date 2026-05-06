SHELL := /bin/bash
STAINLESS_DIR := ../stainless
PYTHON := uv run python
# Auto-detect openapi-generator command (npm installs as openapi-generator-cli, brew as openapi-generator)
OPENAPI_GENERATOR := $(shell command -v openapi-generator-cli 2>/dev/null || command -v openapi-generator 2>/dev/null || echo openapi-generator-cli)

# Package naming: set OPEN=0 to generate as "ogx_client" (without "open")
OPEN ?= 1
ifeq ($(OPEN),0)
  PACKAGE_NAME := ogx_client
  PROJECT_NAME := ogx-client
else
  PACKAGE_NAME := ogx_open_client
  PROJECT_NAME := ogx-open-client
endif

# Scripts
MERGE_SCRIPT := ./merge_stainless_config.py
HIERARCHY_SCRIPT := ./build_hierarchy.py
PATCH_SCRIPT := ./patch_hierarchy.py

# Inputs
PATCHES_FILE := ./patches.yml
STAINLESS_CONFIG := $(STAINLESS_DIR)/config.yml
STAINLESS_OPENAPI := $(STAINLESS_DIR)/openapi.yml
OPENAPI_CONFIG_TEMPLATE := openapi-config.json.template
OPENAPI_CONFIG := openapi-config.json
TEMPLATE_DIR := ./templates/python

# Outputs
OPENAPI_OUTPUT := openapi.yml
PROCESSED_SPEC := openapi-hierarchical.yml
HIERARCHY_FILE := api-hierarchy.yml
SDK_OUTPUT_DIR := sdks/python

# Extract version from root pyproject.toml
VERSION := $(shell grep 'fallback_version' ../../pyproject.toml | cut -d'"' -f2)

.PHONY: all sdk openapi hierarchy clean help version check-generator generate-config

all: sdk

help:
	@echo "Available targets:"
	@echo "  all        - Generate SDK (default)"
	@echo "  openapi    - Generate enriched OpenAPI spec from Stainless config"
	@echo "  hierarchy  - Process spec for hierarchical SDK (leaf tags, dummy endpoints)"
	@echo "  sdk        - Generate Python SDK with hierarchical API structure"
	@echo "  version    - Show version that will be used for SDK"
	@echo "  clean      - Remove generated files"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Pipeline: openapi -> hierarchy -> sdk (each target triggers its dependencies)"
	@echo ""
	@echo "Options:"
	@echo "  OPEN=1     - (default) Generate as ogx_open_client"
	@echo "  OPEN=0     - Generate as ogx_client (without 'open')"

version:
	@echo "SDK version: $(VERSION)"

check-generator:
	@if ! command -v openapi-generator-cli >/dev/null 2>&1 && ! command -v openapi-generator >/dev/null 2>&1; then \
		echo "Error: openapi-generator not found in PATH"; \
		echo ""; \
		echo "Please install it first:"; \
		echo ""; \
		echo "macOS:"; \
		echo "  brew install openapi-generator"; \
		echo "  OR"; \
		echo "  npm install -g @openapitools/openapi-generator-cli"; \
		echo ""; \
		echo "Linux:"; \
		echo "  npm install -g @openapitools/openapi-generator-cli"; \
		echo ""; \
		echo "For more options: https://openapi-generator.tech/docs/installation"; \
		exit 1; \
	fi
	@command -v java >/dev/null 2>&1 || \
		{ echo "Error: java not found in PATH"; \
		  echo ""; \
		  echo "openapi-generator requires Java 11+"; \
		  echo ""; \
		  echo "macOS:"; \
		  echo "  brew install openjdk"; \
		  echo ""; \
		  echo "Linux:"; \
		  echo "  sudo dnf install java-11-openjdk"; \
		  echo "  (or use your package manager: apt, yum, etc.)"; \
		  echo ""; \
		  exit 1; }

# Step 1: Merge Stainless config into OpenAPI spec and apply patches
openapi: $(OPENAPI_OUTPUT)

$(OPENAPI_OUTPUT): $(MERGE_SCRIPT) $(PATCHES_FILE) $(STAINLESS_CONFIG) $(STAINLESS_OPENAPI)
	@echo "Generating OpenAPI spec..."
	$(PYTHON) $(MERGE_SCRIPT) --patch $(PATCHES_FILE) --stainless $(STAINLESS_CONFIG) --openapi $(STAINLESS_OPENAPI) --output $(OPENAPI_OUTPUT)

# Step 2: Extract tag hierarchy, reduce to leaf tags, create dummy endpoints, fix schemas
hierarchy: $(PROCESSED_SPEC)

$(PROCESSED_SPEC): $(HIERARCHY_SCRIPT) $(OPENAPI_OUTPUT)
	@echo "Processing hierarchy..."
	$(PYTHON) $(HIERARCHY_SCRIPT) --source $(OPENAPI_OUTPUT) --output $(PROCESSED_SPEC) --hierarchy $(HIERARCHY_FILE)

# Generate openapi-config.json from template based on OPEN flag
generate-config: $(OPENAPI_CONFIG_TEMPLATE)
	@echo "Generating $(OPENAPI_CONFIG) (OPEN=$(OPEN), package=$(PACKAGE_NAME))..."
ifeq ($(OPEN),0)
	@sed 's/{{open.}}//g' $(OPENAPI_CONFIG_TEMPLATE) > $(OPENAPI_CONFIG)
else
	@sed 's/{{\(open.\)}}/\1/g' $(OPENAPI_CONFIG_TEMPLATE) > $(OPENAPI_CONFIG)
endif

# Step 3: Generate Python SDK from processed spec, then patch in hierarchy
sdk: check-generator $(PROCESSED_SPEC) generate-config
	@echo "Generating Python SDK (version: $(VERSION), package: $(PACKAGE_NAME))..."
	@mkdir -p $(SDK_OUTPUT_DIR)
	$(OPENAPI_GENERATOR) generate -i $(PROCESSED_SPEC) -g python -c $(OPENAPI_CONFIG) -o $(SDK_OUTPUT_DIR) \
		--additional-properties=packageVersion=$(VERSION) -t $(TEMPLATE_DIR)
	@cp -R $(TEMPLATE_DIR)/lib $(SDK_OUTPUT_DIR)/$(PACKAGE_NAME)
	@echo "Patching SDK with hierarchical API structure..."
	$(PYTHON) $(PATCH_SCRIPT) --hierarchy $(HIERARCHY_FILE) --sdk-dir $(SDK_OUTPUT_DIR) --package $(PACKAGE_NAME)

clean:
	@echo "Cleaning generated files..."
	@rm -f $(OPENAPI_OUTPUT) $(PROCESSED_SPEC) $(HIERARCHY_FILE) $(OPENAPI_CONFIG)
	@rm -rf sdks
