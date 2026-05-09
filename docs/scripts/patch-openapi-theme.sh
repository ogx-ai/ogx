#!/bin/bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Patch docusaurus-theme-openapi-docs to hide auto-generated code variants
# (http.client, requests) when x-codeSamples are provided in the OpenAPI spec.
# Only the OpenAI SDK examples will show for endpoints that have x-codeSamples.
FILE="node_modules/docusaurus-theme-openapi-docs/lib/theme/ApiExplorer/CodeSnippets/index.js"
[ ! -f "$FILE" ] && exit 0

python3 -c "
with open('$FILE') as f:
    c = f.read()
# Skip variants tab when x-codeSamples (samples) exist for this language
old = '          react_1.default.createElement(\n            CodeTabs_1.default,\n            {\n              className: \"openapi-tabs__code-container-inner\",\n              action: {\n                setLanguage: setLanguage,\n                setSelectedVariant: setSelectedVariant,\n              },\n              includeVariant: true,'
new = '          !lang.samples && react_1.default.createElement(\n            CodeTabs_1.default,\n            {\n              className: \"openapi-tabs__code-container-inner\",\n              action: {\n                setLanguage: setLanguage,\n                setSelectedVariant: setSelectedVariant,\n              },\n              includeVariant: true,'
if '!lang.samples &&' not in c:
    c = c.replace(old, new)
    with open('$FILE', 'w') as f:
        f.write(c)
    print('Patched: x-codeSamples now replace auto-generated variants')
else:
    print('Already patched')
"
