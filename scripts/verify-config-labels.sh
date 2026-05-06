#!/usr/bin/env bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Verify that config labels are properly embedded in an image
set -euo pipefail

IMAGE_NAME="${1:-}"

if [ -z "$IMAGE_NAME" ]; then
    echo "Usage: $0 IMAGE_NAME" >&2
    exit 1
fi

echo "Verifying config labels in: $IMAGE_NAME"
echo

# Extract ogx labels
if ! docker inspect "$IMAGE_NAME" &>/dev/null; then
    echo "❌ Image not found: $IMAGE_NAME" >&2
    exit 1
fi

labels=$(docker inspect "$IMAGE_NAME" --format='{{ range $k, $v := .Config.Labels }}{{ $k }}={{ $v }}{{ "\n" }}{{ end }}' | grep '^com\.ogx')

if [ -z "$labels" ]; then
    echo "❌ No ogx labels found"
    exit 1
fi

# Extract and validate required metadata labels
distro_name=$(echo "$labels" | grep "^com.ogx.distribution.name=" | cut -d= -f2-)
if [ -z "$distro_name" ]; then
    echo "❌ Missing or empty label: com.ogx.distribution.name"
    exit 1
fi

version=$(echo "$labels" | grep "^com.ogx.distribution.version=" | cut -d= -f2-)
if [ -z "$version" ]; then
    echo "❌ Missing or empty label: com.ogx.distribution.version"
    exit 1
fi

default_config=$(echo "$labels" | grep "^com.ogx.distribution.default-config=" | cut -d= -f2-)
if [ -z "$default_config" ]; then
    echo "❌ Missing or empty label: com.ogx.distribution.default-config"
    exit 1
fi

config_list=$(echo "$labels" | grep "^com.ogx.distribution.configs=" | cut -d= -f2-)
if [ -z "$config_list" ]; then
    echo "❌ Missing or empty label: com.ogx.distribution.configs"
    exit 1
fi

echo "Distribution: $distro_name"
echo "Version: $version"
echo "Default config: $default_config"
echo "Available configs: $config_list"
echo

echo "✅ Required metadata labels present"
echo

# Validate default config is in the configs list
if ! echo "$config_list" | grep -qE "(^|,)${default_config}(,|$)"; then
    echo "❌ Default config '$default_config' not found in configs list: $config_list"
    exit 1
fi

# Validate each config in the list has a corresponding label
IFS=',' read -ra CONFIGS <<< "$config_list"
for config in "${CONFIGS[@]}"; do
    echo "🔍 Config '$config'"

    label_key="com.ogx.config.${config}"
    encoded=$(docker inspect "$IMAGE_NAME" --format="{{ index .Config.Labels \"$label_key\" }}")

    if [ -z "$encoded" ]; then
        echo "❌ Config label not found: $label_key"
        exit 1
    fi

    # Decode base64 to temp file (avoids command substitution stripping trailing newlines)
    decoded_file=$(mktemp)
    if ! echo "$encoded" | base64 -d > "$decoded_file" 2>/dev/null; then
        echo "❌ Failed to decode config: $config"
        rm -f "$decoded_file"
        exit 1
    fi

    # Validate YAML syntax if Python and PyYAML available
    PYTHON_CMD=""
    if command -v python3 &>/dev/null && python3 -c 'import yaml' 2>/dev/null; then
        PYTHON_CMD="python3"
    elif [ -x ".venv/bin/python3" ] && .venv/bin/python3 -c 'import yaml' 2>/dev/null; then
        PYTHON_CMD=".venv/bin/python3"
    fi

    if [ -n "$PYTHON_CMD" ]; then
        if $PYTHON_CMD -c "import yaml; yaml.safe_load(open('$decoded_file'))" 2>/dev/null; then
            echo "   ✅ Valid YAML syntax"
        else
            echo "   ❌ Invalid YAML syntax"
            rm -f "$decoded_file"
            exit 1
        fi
    fi

    # Compare with source file
    source_file="src/ogx/distributions/${distro_name}/${config}"
    if [ ! -f "$source_file" ]; then
        echo "   ⚠️  Source file not found: $source_file (skipping comparison)"
        rm -f "$decoded_file"
        continue
    fi

    if ! diff -q "$decoded_file" "$source_file" >/dev/null 2>&1; then
        echo "   ❌ Does NOT match source file: $source_file"
        echo "   Showing diff:"
        diff "$decoded_file" "$source_file" || true
        rm -f "$decoded_file"
        exit 1
    fi

    echo "   ✅ Matches source file"
    rm -f "$decoded_file"
done

echo
echo "✅ All config labels verified successfully"
