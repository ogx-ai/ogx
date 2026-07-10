# OGX CI

OGX uses GitHub Actions for Continuous Integration (CI). Below is a table detailing what CI the project includes and the purpose.

| Name | File | Purpose |
| ---- | ---- | ------- |
| Backward Compatibility Check | [backward-compat.yml](backward-compat.yml) | Check backward compatibility for config.yaml files |
| Build Distribution Images | [build-distributions.yml](build-distributions.yml) | Build Distribution Images |
| CI Status | [ci-status.yml](ci-status.yml) | Aggregate CI check status |
| CodeQL Workflow Security Scan | [codeql.yml](codeql.yml) | CodeQL Workflow Security Scan |
| Commit Constraint Updates | [commit-constraint-updates.yml](commit-constraint-updates.yml) | Commit Constraint Updates |
| Commit Recordings | [commit-recordings.yml](commit-recordings.yml) | Commit Recordings |
| Dependabot constraint-dependencies | [dependabot-constraints.yml](dependabot-constraints.yml) | Update constraint-dependencies for Dependabot PR |
| Documentation Build | [docs-build.yml](docs-build.yml) | Build and validate documentation |
| File Processors Tests | [file-processors-tests.yml](file-processors-tests.yml) | Run file processors integration tests |
| Installer CI | [install-script-ci.yml](install-script-ci.yml) | Test the installation script |
| Integration Auth Tests | [integration-auth-tests.yml](integration-auth-tests.yml) | Run the integration test suite with Kubernetes authentication |
| Integration Responses, Conversations & Prompts Auth Tests | [integration-responses-conversations-auth-tests.yml](integration-responses-conversations-auth-tests.yml) | Run responses, conversations, and prompts auth tests with Kubernetes authentication |
| SqlStore Integration Tests | [integration-sql-store-tests.yml](integration-sql-store-tests.yml) | Run the integration test suite with SqlStore |
| Responses API - Codex CLI Smoke Test | [integration-tests-codex-cli.yml](integration-tests-codex-cli.yml) | Manually drive the real Codex CLI against /v1/responses |
| Messages API - Claude Code Client Smoke Tests | [integration-tests-messages-clients.yml](integration-tests-messages-clients.yml) | Drive the Claude Code CLI and Agent SDK against /v1/messages (live, Ollama) |
| Integration Tests (Replay) | [integration-tests.yml](integration-tests.yml) | Run the integration test suites from tests/integration in replay mode |
| Vector IO Integration Tests | [integration-vector-io-tests.yml](integration-vector-io-tests.yml) | Run the integration test suite with various VectorIO providers |
| Launch GPU EC2 Runner | [launch-gpu-ec2-runner.yml](launch-gpu-ec2-runner.yml) | GPU recording for gpt-oss:20b (${{ inputs.suite }} suite) |
| OpenAPI Generator SDK Validation | [openapi-generator-validation.yml](openapi-generator-validation.yml) | Validate OpenAPI Generator SDK generation |
| OpenResponses Conformance Tests | [openresponses-conformance.yml](openresponses-conformance.yml) | Run OpenResponses conformance tests against ogx Responses API |
| Post-release automation | [post-release.yml](post-release.yml) | Post-release automation |
| Pre-commit | [pre-commit.yml](pre-commit.yml) | Run pre-commit checks |
| Prepare release | [prepare-release.yml](prepare-release.yml) | Prepare release |
| Test OGX Build | [providers-build.yml](providers-build.yml) | Test ogx build and list-deps |
| Build, test, and publish packages | [pypi.yml](pypi.yml) | Build, test, and publish packages |
| Integration Tests (Record) | [record-integration-tests.yml](record-integration-tests.yml) | Auto-record missing test recordings for PR |
| vLLM GPU Recording | [record-vllm-gpu-tests.yml](record-vllm-gpu-tests.yml) | GPU recording for gpt-oss:20b (${{ inputs.suite }} suite) |
| Release Branch Scheduled CI | [release-branch-scheduled-ci.yml](release-branch-scheduled-ci.yml) | Scheduled CI checks for active release branches |
| Check semantic PR titles | [semantic-pr.yml](semantic-pr.yml) | Ensure that PR titles follow the conventional commit spec |
| Close stale issues and PRs | [stale_bot.yml](stale_bot.yml) | Run the Stale Bot action |
| Test External Providers Installed via Module | [test-external-provider-module.yml](test-external-provider-module.yml) | Test External Provider installation via Python module |
| Test External API and Providers | [test-external.yml](test-external.yml) | Test the External API and Provider mechanisms |
| Trigger Docs Deploy | [trigger-docs-deploy.yml](trigger-docs-deploy.yml) | Trigger docs site rebuild after docs change |
| Trivy Scheduled Security Scan | [trivy-scheduled.yml](trivy-scheduled.yml) | Trivy Scheduled Security Scan |
| Trivy Security Scan | [trivy-security.yml](trivy-security.yml) | Trivy Security Scan |
| UI Tests | [ui-unit-tests.yml](ui-unit-tests.yml) | Run the UI test suite |
| Unit Tests | [unit-tests.yml](unit-tests.yml) | Run the unit test suite |
