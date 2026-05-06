# Release Notes Generation with Claude Code

This document contains a self-contained prompt for generating release notes using [Claude Code](https://docs.anthropic.com/en/docs/claude-code). It produces two artifacts:

1. **Detailed release notes** in `docs/releases/RELEASE_NOTES_{VERSION}.md`
2. **Summary section** prepended to the top-level `RELEASE_NOTES.md`
3. **Pull request** with the changes

## Usage

Run from the repository root, providing the version range as a prefix:

```bash
(echo "OLD_VERSION=0.4.0 NEW_VERSION=0.5.0" && cat docs/releases/GENERATE.md) | claude -p -
```

Claude will resolve `{OLD_VERSION}` and `{NEW_VERSION}` throughout the prompt from these definitions.

## Prompt

Create release notes between version {OLD_VERSION} and {NEW_VERSION} for this project.

### Step 1: Analyze Changes

1. Find git tags for {OLD_VERSION} and {NEW_VERSION} (or HEAD if not yet tagged)
2. Get ALL commits between these versions
3. For commits marked with `!` in conventional commits (e.g., `feat!:`, `fix!:`), examine the actual diffs
4. Also check unmarked commits for breaking changes by reviewing the code changes themselves. Do your due diligence: look for non-compatible API schema changes, changes of default values, removed features, renamed config fields, changed method signatures, etc.

### Step 2: Classify Breaking Changes

Separate breaking changes into THREE categories:

**Hard Breaking Changes** (action required before upgrading):
Changes where the user HAS to update something or their code/config will fail. Only include truly hard breaks here, not things that are backwards compatible.

**Deprecated** (works with warnings, migrate before next major release):
Changes that include a backward compatibility layer with deprecation warnings.

**Behavior Changes** (no code changes required, but be aware):
Changes to default values or response formats where existing code still works but behaves differently.

### Step 3: Write Detailed Release Notes

Write the detailed release notes to `docs/releases/RELEASE_NOTES_{NEW_VERSION}.md` using this structure:

1. **Title**: `# OGX {NEW_VERSION} Release Notes`
2. **Release Date**: Use the date from the GitHub release tag, or today if not yet released
3. **One-paragraph summary** of the release highlights
4. **Breaking Changes** section with:
   - Summary tables for each category (Hard Breaking, Deprecated, Behavior Changes)
   - Each table has columns: Change | Migration/Note | PR link
   - A disclaimer that breaking changes are discussed in detail below
   - Detailed subsections for each breaking change with:
     - PR link and contributor full name with company (fetch from GitHub if needed)
     - Impact description (who is affected)
     - Before/After code examples where applicable
     - Migration instructions
5. **New Features** with subsections for major features, grouped logically
6. **New Providers** if any
7. **API/Architecture Changes** (e.g., migration patterns)
8. **Backward Compatibility Improvements**
9. **Bug Fixes** (one-liner list with PR links and contributor names)
10. **Documentation** updates
11. **Upgrade Guide** with:
    - "Before Upgrading" section for hard breaking changes (step-by-step with grep commands to find affected code)
    - "After Upgrading" section for deprecations

Formatting rules for the detailed notes:

- All PR references link to GitHub: `[#NNNN](https://github.com/ogx-ai/ogx/pull/NNNN)`
- For each change, fetch the PR author's full name from GitHub. Include company affiliation in parentheses.
- Do NOT put author names in the summary overview tables, only in the detailed sections.
- Use code blocks with language hints for before/after examples.

### Step 4: Update Top-Level RELEASE_NOTES.md

Prepend a summary section to `RELEASE_NOTES.md`, inserting it directly below the HTML comment marker (`<!-- New releases go here ...`). Use this structure:

```markdown
## [{NEW_VERSION}](docs/releases/RELEASE_NOTES_{NEW_VERSION}.md) - {RELEASE_DATE}

{One-paragraph summary of the release.}

### Highlights

- **Feature 1** short description ([#NNNN](link))
- **Feature 2** short description ([#NNNN](link))
- ...

### Breaking Changes

| Change | Type | PR |
|--------|------|-----|
| Short description of change | Hard/Deprecated | [#NNNN](link) |
| ... | ... | ... |

See the [full release notes](docs/releases/RELEASE_NOTES_{NEW_VERSION}.md) for migration instructions and detailed upgrade guide.
```

Guidelines for the summary:

- Use ISO date format (`YYYY-MM-DD`) from the GitHub release
- List 5-10 of the most notable features in Highlights (not exhaustive)
- The Breaking Changes table lists all breaking and deprecated changes with their type (`Hard` or `Deprecated`), but without migration details
- Always prepend new releases (newest at the top)

### Step 5: Create Pull Request

1. Create a new branch `docs/release-notes-{NEW_VERSION}`
2. Commit both files with message: `docs: add release notes for version {NEW_VERSION}`
3. Push and create a PR against `main` with:
   - Title: `docs: add release notes for version {NEW_VERSION}`
   - Body summarizing what's included (number of breaking changes, features, bug fixes, etc.)
