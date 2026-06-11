#!/bin/bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Patches for docusaurus-theme-openapi-docs to support multiple SDK code
# samples (OpenAI, Anthropic, Google) as separate language tabs.

SNIPPETS="node_modules/docusaurus-theme-openapi-docs/lib/theme/ApiExplorer/CodeSnippets/index.js"
TABS="node_modules/docusaurus-theme-openapi-docs/lib/theme/ApiExplorer/CodeTabs/index.js"
[ ! -f "$SNIPPETS" ] && exit 0

# 1. Guard the variants section: skip when x-codeSamples exist OR when
#    the language has no postman variants (custom SDK tabs).
python3 -c "
with open('$SNIPPETS') as f:
    c = f.read()
old = '          react_1.default.createElement(\n            CodeTabs_1.default,\n            {\n              className: \"openapi-tabs__code-container-inner\",\n              action: {\n                setLanguage: setLanguage,\n                setSelectedVariant: setSelectedVariant,\n              },\n              includeVariant: true,'
new = '          !lang.samples && lang.variants && react_1.default.createElement(\n            CodeTabs_1.default,\n            {\n              className: \"openapi-tabs__code-container-inner\",\n              action: {\n                setLanguage: setLanguage,\n                setSelectedVariant: setSelectedVariant,\n              },\n              includeVariant: true,'
if 'lang.variants &&' not in c:
    c = c.replace(old, new)
    with open('$SNIPPETS', 'w') as f:
        f.write(c)
    print('Patched: guard variants rendering')
else:
    print('Already patched (variants guard)')
"

# 2. Hide language tabs that have no content (no x-codeSamples and no
#    postman variants). Custom SDK tabs only appear on endpoints that
#    include their code samples.
python3 -c "
with open('$SNIPPETS') as f:
    c = f.read()
old = '      mergedLangs.map((lang) => {'
new = '      mergedLangs.filter((lang) => lang.samples || (lang.variants && lang.variants.length > 0)).map((lang) => {'
if '.filter((lang) => lang.samples' not in c:
    c = c.replace(old, new)
    with open('$SNIPPETS', 'w') as f:
        f.write(c)
    print('Patched: hide empty language tabs')
else:
    print('Already patched (empty tab filter)')
"

# 3. Fix CodeTabs crash when clicking a language tab without postman
#    variants (custom SDK tabs). The click handler assumes all languages
#    have a variants array, which is not true for custom entries.
python3 -c "
with open('$TABS') as f:
    c = f.read()
old = '''        newLanguage = languageSet.filter(
          (lang) => lang.language === newTabValue
        )[0];
        action.setSelectedVariant(newLanguage.variants[0].toLowerCase());
        action.setSelectedSample(newLanguage.sample);'''
new = '''        newLanguage = languageSet.filter(
          (lang) => lang.language === newTabValue
        )[0];
        if (newLanguage.variants) { action.setSelectedVariant(newLanguage.variants[0].toLowerCase()); }
        action.setSelectedSample(newLanguage.sample);'''
if 'if (newLanguage.variants)' not in c:
    c = c.replace(old, new)
    with open('$TABS', 'w') as f:
        f.write(c)
    print('Patched: guard variants access in tab click handler')
else:
    print('Already patched (tab click guard)')
"

# 4. When a language tab has exactly one code sample, render the code
#    block directly instead of wrapping it in a redundant inner tab.
python3 -c "
with open('$SNIPPETS') as f:
    c = f.read()
old = '''          lang.samples &&
            react_1.default.createElement(
              CodeTabs_1.default,
              {
                className: \"openapi-tabs__code-container-inner\",'''
new = '''          lang.samples && lang.samples.length === 1 &&
            react_1.default.createElement(
              ApiCodeBlock_1.default,
              {
                language: lang.highlight,
                className: \"openapi-explorer__code-block\",
                showLineNumbers: true,
              },
              codeSampleCodeText
            ),
          lang.samples && lang.samples.length > 1 &&
            react_1.default.createElement(
              CodeTabs_1.default,
              {
                className: \"openapi-tabs__code-container-inner\",'''
if 'lang.samples.length === 1' not in c:
    c = c.replace(old, new)
    with open('$SNIPPETS', 'w') as f:
        f.write(c)
    print('Patched: skip inner tab for single code sample')
else:
    print('Already patched (single sample shortcut)')
"

# 5. Fix defaultValue crash on endpoints without SDK code samples.
#    The default picks the first entry in mergedLangs (e.g. "OpenAI"),
#    but after filtering only "curl" may remain, causing a mismatch.
python3 -c "
with open('$SNIPPETS') as f:
    c = f.read()
old = 'defaultValue: defaultLang[0]?.language ?? mergedLangs[0].language,'
new = 'defaultValue: defaultLang[0]?.language ?? (mergedLangs.find((l) => l.samples || (l.variants && l.variants.length > 0)) || mergedLangs[0])?.language,'
if 'mergedLangs.find' not in c:
    c = c.replace(old, new)
    with open('$SNIPPETS', 'w') as f:
        f.write(c)
    print('Patched: defaultValue picks first visible tab')
else:
    print('Already patched (defaultValue)')
"

# 6. Fix initial language state on endpoints without SDK code samples.
#    The language state defaults to mergedLangs[0] (e.g. "OpenAI") but
#    only "curl" may be visible, so curl code never gets generated.
python3 -c "
with open('$SNIPPETS') as f:
    c = f.read()
old = '''  const [language, setLanguage] = (0, react_1.useState)(() => {
    // Return first index if only 1 user-defined language exists
    if (mergedLangs.length === 1) {
      return mergedLangs[0];
    }
    // Fall back to language in localStorage or first user-defined language
    return defaultLang[0] ?? mergedLangs[0];
  });'''
new = '''  const _firstVisible = mergedLangs.find((l) => l.samples || (l.variants && l.variants.length > 0)) || mergedLangs[0];
  const [language, setLanguage] = (0, react_1.useState)(() => {
    // Return first index if only 1 user-defined language exists
    if (mergedLangs.length === 1) {
      return mergedLangs[0];
    }
    // Fall back to language in localStorage or first visible language
    return defaultLang[0] ?? _firstVisible;
  });'''
if '_firstVisible' not in c:
    c = c.replace(old, new)
    with open('$SNIPPETS', 'w') as f:
        f.write(c)
    print('Patched: initial language picks first visible tab')
else:
    print('Already patched (initial language)')
"
