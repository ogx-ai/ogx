# Deployment Guide for ogx-open-client

This document outlines the deployment configuration and requirements for publishing the `ogx-open-client` SDK to PyPI.

## Required GitHub Configurations

### 1. Environments

The publishing workflow requires two GitHub environments with different protection rules:

#### `testpypi` Environment

- **Purpose**: For testing releases and automated tag-triggered publishes
- **Protection rules**: None (auto-approve)
- **Secrets**:
  - `TEST_PYPI_API_TOKEN` — TestPyPI API token

#### `pypi-production` Environment

- **Purpose**: For production releases to PyPI
- **Protection rules**: **Required reviewers** (at least 1)
- **Secrets**:
  - `PYPI_API_TOKEN` — Production PyPI API token
- **Deployment branches**: Limit to `main` or specific release branches

**Setup Instructions:**

1. Go to Repository Settings → Environments
2. Create `testpypi` environment (no protection)
3. Create `pypi-production` environment
4. Add required reviewers: @ashwinb, @leseb, or @bbrowning
5. Configure deployment branch restrictions (recommended: `main` only)

### 2. Secrets

Configure the following secrets at Repository Settings → Secrets and variables → Actions:

| Secret Name | Purpose | How to Get |
|-------------|---------|-----------|
| `TEST_PYPI_API_TOKEN` | TestPyPI publishing | [Create token at test.pypi.org](https://test.pypi.org/manage/account/token/) |
| `PYPI_API_TOKEN` | Production PyPI publishing | [Create token at pypi.org](https://pypi.org/manage/account/token/) |

**Token Scope**: Set to "Entire account" or limit to `ogx-open-client` package once it exists.

**Security Notes**:

- Use token authentication (not username/password)
- Enable 2FA on PyPI accounts
- Rotate tokens annually
- Prefer Trusted Publishing (OIDC) when available

### 3. PyPI Package Setup

Before first publish, claim the package name:

1. **Register package on TestPyPI first**:

   ```bash
   # Build locally
   cd client-sdks/openapi
   make sdk OPEN=1
   cd sdks/python
   uv build

   # Publish to TestPyPI
   uv publish --publish-url https://test.pypi.org/legacy/ dist/*
   ```

2. **Verify on TestPyPI**: <https://test.pypi.org/project/ogx-open-client/>

3. **Register on production PyPI**:
   - Use workflow_dispatch with `publish_to: pypi`
   - Requires approval from configured reviewers

4. **Configure PyPI project** (post-first-publish):
   - Add project description/README
   - Configure Trusted Publishing (GitHub Actions OIDC)
   - Add maintainers: <contributors@ogx.dev>

## Publishing Workflows

### Automatic Publishing (Tag-Triggered)

Stable releases automatically publish to TestPyPI:

```bash
# Tag format: openapi-sdk-v{VERSION}
git tag openapi-sdk-v1.0.0
git push origin openapi-sdk-v1.0.0
```

**What happens:**

1. Workflow triggers on tag push
2. Builds SDK from OpenAPI spec
3. Publishes to **TestPyPI** (testpypi environment)
4. Uploads artifacts to GitHub

**Pre-release tags** (`-rc`, `-alpha`, `-beta`) are built but **not published**.

### Manual Publishing (Workflow Dispatch)

For production PyPI or on-demand builds:

1. Go to Actions → "Publish OpenAPI SDK to PyPI"
2. Click "Run workflow"
3. Configure:
   - **publish_to**: `testpypi` or `pypi`
   - **dry_run**: `true` (build only) or `false` (publish)

**Production publish** (`publish_to: pypi`):

- Requires approval from environment reviewers
- Notifies reviewers via GitHub
- Reviewer approves/rejects in Actions tab

### Manual Local Publishing

For emergency releases or testing:

```bash
# Generate SDK
cd client-sdks/openapi
make sdk OPEN=1

# Build package
cd sdks/python
uv build

# Publish to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/ dist/*
# Token: paste TEST_PYPI_API_TOKEN when prompted

# Publish to PyPI (production)
uv publish dist/*
# Token: paste PYPI_API_TOKEN when prompted
```

## Versioning

SDK versions follow semantic versioning and track OGX server versions:

- **Stable**: `1.0.0`, `1.1.0`, `1.2.3`
- **Pre-release**: `1.0.0-rc1`, `1.1.0-alpha.2`, `1.2.0-beta.1`

Version is extracted from `../../pyproject.toml` (`fallback_version` field).

To release a new version:

```bash
# Update version in root pyproject.toml
vim pyproject.toml  # Change fallback_version = "1.1.0"

# Commit version bump
git add pyproject.toml
git commit -s -m "chore: bump version to 1.1.0"

# Tag and push
git tag openapi-sdk-v1.1.0
git push origin main
git push origin openapi-sdk-v1.1.0
```

## Rollback Procedure

If a bad version is published:

### 1. Yank the Release

```bash
# TestPyPI
uv publish --yank ogx-open-client==1.0.0 --publish-url https://test.pypi.org/legacy/

# Production PyPI
uv publish --yank ogx-open-client==1.0.0
```

**Effect**: Marks release as unavailable for new installs. Existing installs unaffected.

### 2. Publish Hotfix

```bash
# Fix the issue in code
git commit -s -m "fix: critical bug in 1.0.0"

# Bump to patch version
vim pyproject.toml  # Change to 1.0.1

# Tag and publish
git tag openapi-sdk-v1.0.1
git push origin openapi-sdk-v1.0.1
```

### 3. Notify Users

- Create GitHub Release with changelog
- Post to Discord #announcements
- Update PyPI description if critical security issue

## Monitoring

### Success Indicators

- ✅ Workflow completes with green checkmark
- ✅ Package appears on PyPI/TestPyPI
- ✅ Installation works: `pip install ogx-open-client`
- ✅ Import works: `python -c "from ogx_open_client import OgxClient"`

### Failure Scenarios

| Error | Cause | Solution |
|-------|-------|----------|
| `HTTP 403: Invalid or non-existent authentication` | Bad API token | Rotate secret, update in GitHub |
| `HTTP 400: File already exists` | Version already published | Bump version, cannot overwrite |
| `No such file or directory: dist/` | Build failed | Check `make sdk` step logs |
| `ModuleNotFoundError: ogx_open_client` | Package name typo | Check `packageName` in openapi-config.json |
| Environment approval timeout | No reviewer available | Add more reviewers to environment |

### Debug Workflow

```bash
# Run workflow with dry-run to test build without publishing
# GitHub Actions → Publish OpenAPI SDK to PyPI → Run workflow
# Set: dry_run = true
```

## Security Considerations

1. **Token Rotation**: Rotate PyPI tokens annually
2. **2FA Enforcement**: All PyPI account owners must use 2FA
3. **Environment Protection**: Production requires reviewer approval
4. **Audit Trail**: All publishes logged in GitHub Actions
5. **Package Signing**: Consider adding GPG signatures (future enhancement)

## Trusted Publishing (Future)

Once the package exists on PyPI, migrate to Trusted Publishing:

1. PyPI Project Settings → Publishing
2. Add GitHub publisher:
   - Owner: `ogx-ai`
   - Repository: `ogx`
   - Workflow: `publish-openapi-sdk.yml`
   - Environment: `pypi-production`

3. Remove `PYPI_API_TOKEN` secret (no longer needed)

**Benefits**:

- No long-lived API tokens
- OIDC-based authentication
- Automatic token rotation
- Better security posture

## Troubleshooting

### Package Not Found After Publish

Wait 5-10 minutes for PyPI's CDN to propagate. Then:

```bash
pip install --upgrade --force-reinstall ogx-open-client
```

### Version Conflict

If local environment has old version cached:

```bash
pip cache purge
pip install ogx-open-client==1.0.0  # Specify exact version
```

### Build Artifacts Missing

If build step succeeds but publish fails:

1. Download artifacts from workflow run
2. Manually publish: `uv publish path/to/downloaded/dist/*`

## Contacts

- **Primary Maintainers**: @ashwinb, @leseb, @bbrowning
- **Emergency Contact**: <contributors@ogx.dev>
- **GitHub Issues**: <https://github.com/ogx-ai/ogx/issues>

---

Last updated: 2026-05-31
