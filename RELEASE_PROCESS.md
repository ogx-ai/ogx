# OGX Release Process

This document outlines the release process for OGX, providing predictability for the community on feature delivery timelines and release expectations.

## Release Schedule

OGX follows [Semantic Versioning](https://semver.org/) with three release streams:

| Release Type | Cadence | Description |
|-------------|---------|-------------|
| **Major (X.0.0)** | Every 6-8 months | Breaking changes, major new features, architectural changes |
| **Minor (0.Y.0)** | Monthly | New features, non-breaking API additions, significant improvements |
| **Patch (0.0.Z)** | Weekly | Bug fixes, security patches, documentation updates |

## Version Numbering

Releases follow the `X.Y.Z` pattern:

- **X (Major)**: Incremented for breaking changes or significant architectural updates
- **Y (Minor)**: Incremented for new features and non-breaking enhancements
- **Z (Patch)**: Incremented for bug fixes and minor improvements

### Release Candidates

For minor and major releases, release candidates (RC) are published before the final release:

- Format: `vX.Y.ZrcN` (e.g., `v0.4.0rc1`, `v0.4.0rc2`)
- Python RC packages are published to test.pypi for community testing
- Multiple RCs may be issued until the release is stable

## Branching Strategy

- **`main`**: Active development branch, always contains the latest code
- **`release-X.Y.x`**: Release branches for each minor version (e.g., `release-0.3.x`, `release-0.4.x`)
- Patch releases are made from release branches
- Critical fixes are backported from `main` to active release branches using Mergify

## Milestone Management

### Tracking Work

- **Issues only**: Add only issues to milestones, not PRs (avoids duplicate tracking)
- **Milestone creation**: Create milestones for each planned minor and major release
- **Small fixes**: Quick-landing PRs for small fixes don't require milestone tracking

### Release Criteria

A version is released when:

1. All issues in the corresponding milestone are completed, **OR**
2. Remaining issues are moved to a future milestone with documented rationale

### Triaging

- Triagers manage milestones and prioritize issues
- Discussions happen in the `#triage` Slack channel
- Priority decisions are reviewed in community calls

## Release Process

### Release Owner

Each release has a designated **Release Owner** from the [CODEOWNERS](./CODEOWNERS) group who is responsible for:

1. Creating a dedicated Slack thread in `#release` channel
2. Coordinating testing activities
3. Managing the release timeline
4. Publishing release artifacts
5. Announcing the release

### Testing Requirements

Testing requirements scale with release type:

#### Patch Releases (Z-stream)

- Rely primarily on automated CI tests
- Quick turnaround for critical fixes
- Manual verification only for specific fix validation

#### Minor Releases (Y-stream)

- Automated CI tests must pass
- Manual feature testing for new functionality
- Documentation verification
- **Community testing window: 1 week**
- Release candidates published for community validation

#### Major Releases (X-stream)

- Comprehensive automated test suite
- Scheduled testing period with predefined test plans
- Cross-provider compatibility testing
- Performance benchmarking
- **Community testing window: 2-3 weeks**
- Multiple release candidates as needed

### Release Checklist

For each release, the Release Owner should complete:

- [ ] Create release-specific thread in `#releases` Slack channel
- [ ] Complete the technical release steps below
- [ ] Generate release notes
- [ ] Announce in `#announcements` Slack channel

### Technical Release Steps

#### Patch release (e.g., 0.4.5 on existing `release-0.4.x`)

**Pre-release on `release-0.4.x`:**

Backports are handled automatically by Mergify — patch releases ship whatever has already been backported to the release branch. No manual cherry-picking needed.

- [ ] Wait for the release branch's required checks to pass.
- [ ] Run [**Build, test, and publish packages**](https://github.com/ogx-ai/ogx/actions/workflows/pypi.yml) from `release-0.4.x` with `version=0.4.5`, `packages=clients-only`, and `dry_run=build-only`.
- [ ] After the client build passes, run the same workflow from the same release branch with `dry_run=off` to publish the clients at exactly `0.4.5`.
  - Keep the default client ref so external clients use their matching release branch.
  - Set `skip_latest=true` when releasing an older maintenance line.
  - Verify `ogx-client==0.4.5` is available on PyPI and npm before preparing the server release. Do not substitute a different client version if publication fails.
- [ ] Run the [**Prepare release**](https://github.com/ogx-ai/ogx/actions/workflows/prepare-release.yml) workflow:
  - Input `version`: `0.4.5`
  - Input `release_branch`: `release-0.4.x`
  - This verifies the matching client is on PyPI, updates `fallback_version` and the exact `ogx-client` pin, regenerates `uv.lock`, and runs `uv lock --check` before committing all three files to the release branch.
- [ ] Verify the preparation commit and its required checks pass before tagging. A failed client lookup or dependency resolution leaves the release branch unmodified.

**Release:**

- [ ] Create GitHub release: tag `v0.4.5`, target `release-0.4.x`
  - The release workflow publishes the server packages using the clients published above.
  - Mark an older maintenance release as not latest.
- [ ] Verify all 4 packages published:
  - [ogx on PyPI](https://pypi.org/project/ogx/)
  - [ogx-api on PyPI](https://pypi.org/project/ogx-api/)
  - [ogx-client on PyPI](https://pypi.org/project/ogx-client/)
  - [ogx-client on npm](https://www.npmjs.com/package/ogx-client)

**Post-release (automated):**

The following steps are handled automatically by the [**Post-release automation**](https://github.com/ogx-ai/ogx/actions/workflows/post-release.yml) workflow, which triggers on `release: published`:

- Tags `main` with the next dev tag and opens a fallback-version PR when the release is at least as new as main's current version.
- Opens the npm lockfile update PR for `release-0.4.x`.

#### Minor release (e.g., 0.5.0 — new release branch)

**All of the above, plus:**

- [ ] Create `release-0.5.x` branch off `main`
- [ ] Ensure the release branch has the setuptools-scm config in both `pyproject.toml` files (`dynamic = ["version"]`, `[tool.setuptools_scm]`, etc.)

## Release Artifacts

Each release includes:

- **PyPI package**: `ogx` and `ogx-client`
- **npm package**: `ogx-client`
- **Docker images**: Distribution images on Docker Hub
- **GitHub Release**: Tagged release with release notes
- **Documentation**: Updated docs at <https://ogx-ai.github.io>

See [CONTRIBUTING.md](./CONTRIBUTING.md) for general contribution guidelines.

## Maintenance Policy

OGX actively maintains the **last 2 stable minor releases**.

### What "Maintained" Means

- **Bug fixes**: Critical bugs are backported to maintained release branches
- **Security patches**: Security vulnerabilities are patched in maintained releases
- **Patch releases (Z-stream)**: Maintained releases receive regular patch releases

### Maintenance Timeline

| Release | Status | Notes |
|---------|--------|-------|
| Current minor (0.Y.0) | ✅ Actively maintained | Bug fixes and security patches |
| Previous minor (0.Y-1.0) | ✅ Maintained | Bug fixes and security patches |
| Older releases | ❌ Unmaintained | No backports; upgrade recommended |

### Example

If the current release is `v0.4.x`:

- `v0.4.x` — Actively maintained (current)
- `v0.3.x` — Maintained (bug fixes only)
- `v0.2.x` and earlier — Unmaintained

Users on unmaintained versions are encouraged to upgrade to continue receiving fixes.

## How the Release Workflow Works

The unified workflow (`.github/workflows/pypi.yml`) builds and publishes all packages:

- **Local packages** (ogx, ogx-api): version comes from the git tag via `SETUPTOOLS_SCM_PRETEND_VERSION`
- **External packages** (ogx-client python/typescript): the workflow patches `pyproject.toml`/`_version.py`/`package.json` at build time using the tag version via `sed`/`npm version`
- `fallback_version` is only used for nightly/dev builds and Docker — not for releases
- Manual workflows run from the selected branch or tag; release workflows run from the released tag. Select the target release branch for the client publication step.

### Workflow Modes

| Trigger | Version | Target |
|---|---|---|
| `release: published` | From tag (`v0.4.5` → `0.4.5`) | Server packages on pypi.org; clients must already be published |
| `schedule` (nightly) | `{base}.dev{YYYYMMDD}` (from dev tag or fallback) | test.pypi.org |
| `workflow_dispatch` dry_run=test-pypi | `{base}.dev{YYYYMMDD}` or manual `version` input | test.pypi.org |
| `workflow_dispatch` dry_run=off | Manual `version` input | pypi.org + npm |
| `workflow_dispatch` dry_run=build-only | N/A | No publish |

## Automation Workflows

### Prepare release (`.github/workflows/prepare-release.yml`)

Triggered via `workflow_dispatch` after publishing the matching clients. Takes a version and matching release branch as input, then:

- Verifies `ogx-client==X.Y.Z` is available on PyPI
- Updates `fallback_version` to the release version in both `pyproject.toml` files
- Updates `ogx-client` pins to `==X.Y.Z`
- Regenerates `uv.lock` and verifies it with `uv lock --check`
- Commits both manifests and the lockfile directly to the release branch only after all checks pass

### Post-release (`.github/workflows/post-release.yml`)

Triggered automatically after the `pypi.yml` workflow succeeds for a release event. Handles:

- **Dev tag**: Tags `main` with `vX.Y.(Z+1)-dev` when the released stable version is at least as new as main's current version
- **Fallback bump**: Opens a PR to bump `fallback_version` to the next `.dev0` under the same condition
- **npm lockfile**: Opens a PR to the release branch updating the UI lockfile

Older maintenance releases leave main's version and dev tags unchanged while still updating their release-branch UI dependency.

### Nightly version computation

The nightly build (in `pypi.yml`) derives its base version from `git describe --tags --match 'v*'`, using the dev tag pushed by the post-release workflow. `fallback_version` in `pyproject.toml` serves as a safety net for builds without git history (e.g., source tarballs).

## Future Improvements

### 1. Let setuptools-scm infer version from tags directly

Right now the workflow computes the version separately and passes it via `SETUPTOOLS_SCM_PRETEND_VERSION`. With dev tags now on `main`, setuptools-scm can potentially infer versions natively, which would:

- Eliminate the `compute-version` step entirely
- Eliminate `fallback_version` management (no more bumping it post-release)
- Make `uv build` work correctly locally without any env vars
- Let setuptools-scm generate dev versions automatically (e.g., `0.5.0.dev3+gabcdef` based on commits since last tag)

### 2. Client repos should use dynamic versioning

The `ogx-client-python` and `ogx-client-typescript` repos use static versions. The workflow patches them with `sed` at build time, which is fragile. If those repos adopted setuptools-scm (Python) or a similar scheme, the workflow could just set an env var instead of rewriting files.
