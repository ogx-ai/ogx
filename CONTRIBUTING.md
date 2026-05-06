# Contributing to OGX

We want to make contributing to this project as easy and transparent as
possible.

## Set up your development environment

We use [uv](https://github.com/astral-sh/uv) to manage python dependencies and virtual environments.
You can install `uv` by following this [guide](https://docs.astral.sh/uv/getting-started/installation/).

You can install the dependencies by running:

```bash
cd ogx
uv venv --python 3.12
uv sync --group dev
uv pip install -e .
source .venv/bin/activate
```

**Note:** If you are making changes to OGX, it is essential that you use Python 3.12 as shown above.
OGX can work with Python 3.13 but the pre-commit hooks used to validate code changes only work with Python 3.12.
If you don't specify a Python version, `uv` will automatically select a Python version according to the `requires-python`
section of the `pyproject.toml`, which is fine for running OGX but not for committing changes.
For more info, see the [uv docs around Python versions](https://docs.astral.sh/uv/concepts/python-versions/).

Note that you can create a dotenv file `.env` that includes necessary environment variables:

```text
OGX_BASE_URL=http://localhost:8321
OGX_CLIENT_LOG=debug
OGX_PORT=8321
OGX_CONFIG=server:ci-tests
TAVILY_SEARCH_API_KEY=
BRAVE_SEARCH_API_KEY=
```

And then use this dotenv file when running client SDK tests via the following:

```bash
uv run --env-file .env -- pytest -v tests/integration/conversations/test_openai_conversations.py --setup=gpt
```

For additional information about testing, please refer to the [tests documentation](tests/README.md) for both unit and integration test.

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to run linting and formatting checks on your code. You can install the pre-commit hooks by running:

```bash
uv pip install 'pre-commit>=4.4.0'
uv run pre-commit install
```

Note that the pre-commit version used in continuous integration for OGX is `>=4.4.0`. It is essential that you use the version specifier as shown above. Once you have run these commands, pre-commit hooks will run automatically before each commit.

Alternatively, if you don't want to install the pre-commit hooks (or if you want to check if your changes are ready before committing),
you can run the checks manually by running:

```bash
uv run pre-commit run --all-files -v
```

The `-v` (verbose) parameter is optional but often helpful for getting more information about any issues with that the pre-commit checks identify.

To run the expanded mypy configuration that CI enforces, use:

```bash
uv run pre-commit run mypy-full --hook-stage manual --all-files
```

or invoke mypy directly with all optional dependencies:

```bash
uv run --group dev --group type_checking mypy
```

```text
**Caution:** Before pushing your changes, make sure that the pre-commit hooks have passed successfully.
```

## Issues and Pull Requests

We actively welcome your pull requests. However, please read the following. This is heavily inspired by [Ghostty](https://github.com/ghostty-org/ghostty/blob/main/CONTRIBUTING.md).

If in doubt, please open a [issue](https://github.com/ogx-ai/ogx/issues).

### Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

The OGX Contributors has a [bounty program](http://facebook.com/whitehat/info) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

### Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of The OGX Contributors's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

### I'd like to contribute

If you are new to the project, start by looking at the issues tagged with "good first issue". If you're interested
leave a comment on the issue and a triager will assign it to you.

Please avoid picking up too many issues at once. This helps you stay focused and ensures that others in the community also have opportunities to contribute.

- Try to work on only 1–2 issues at a time, especially if you’re still getting familiar with the codebase.
- Before taking an issue, check if it’s already assigned or being actively discussed.
- If you’re blocked or can’t continue with an issue, feel free to unassign yourself or leave a comment so others can step in.

### I have a bug

1. Search the issue tracker for similar issues.
2. If you don't have steps to reproduce, open a "discussion-type" issue.
3. If you have steps to reproduce, open a "bug-type" issue.

### I have an idea for a feature

1. Open a "discussion-type" issue.

### I've implemented a feature

1. If there is an issue for the feature, open a pull request.
2. If there is no issue, open a "discussion-type" issue and link to your branch.

### I have a question

1. Open a "discussion-type" issue or use [Slack](https://join.slack.com/t/ogx-ai/shared_invite/zt-3uyw5bxj9-tSEwsNZncgkGEKbd4dXIpw).

### Opening a Pull Request

1. Fork the repo and create your branch from `main`.
2. If you've changed APIs, update the documentation.
3. Ensure the test suite passes.
4. Make sure your code lints using `pre-commit`.
5. If you haven't already, complete the Contributor License Agreement ("CLA").
6. Ensure your pull request follows the [conventional commits format](https://www.conventionalcommits.org/en/v1.0.0/).
7. Ensure your pull request follows the [coding style](#coding-style).

Please keep pull requests (PRs) small and focused. If you have a large set of changes, consider splitting them into logically grouped, smaller PRs to facilitate review and testing.

**Tip:** As a general guideline:

- Experienced contributors should try to keep no more than 5 open PRs at a time.
- New contributors are encouraged to have only one open PR at a time until they're familiar with the codebase and process.

### Handling Pull Request Reviews

To ensure a smooth review process for both reviewers and contributors, please follow these guidelines:

1. **Do not amend commits and force push.** Force pushing rewrites history, which confuses GitHub's review system and makes it difficult for reviewers to see what changed between review rounds. The diff becomes unreliable and reviewers lose track of their previous comments.

2. **Use `git merge` to update your branch from main.** When you need to incorporate the latest changes from the main branch, use `git merge main` instead of `git rebase main`. This preserves the commit history and avoids the issues associated with force pushing.

3. **Address review comments with new commits.** When responding to feedback, add new commits rather than modifying existing ones. This makes it easy for reviewers to see exactly how each comment was addressed. You can also use GitHub's "Accept suggestion" feature directly in the UI, which automatically creates new commits for you.

Since OGX uses squash merge, all commits will be combined into a single commit when the PR is merged. This means you don't need to worry about having a "clean" commit history during review; what matters is making the review process as clear as possible.

### API Enhancement Pull Requests

If your pull request modifies or extends the OGX API, you must include a comprehensive **Test Plan** in your PR description. This helps reviewers verify the correctness of your changes and ensures the API behaves as expected.

Your Test Plan should include:

1. **A testing script**: Provide a script (Python, curl commands, or similar attached or pasted into the PR description) that demonstrates your API changes in action. The script should be self-contained and
   easy to run. Do not commit the script to the repository.

2. **Execution output**: Include the actual output from running your test script. This proves your implementation works and gives reviewers a clear picture of the expected behavior.

This practice:

- Validates that your implementation works end-to-end
- Provides reviewers with reproducible verification steps
- Serves as documentation for the new API behavior
- Catches integration issues early in the review process

### Adding a New Provider

OGX has "out-of-tree" providers referred to as [external providers](https://ogx-ai.github.io/docs/providers/external) as well as "in-tree" providers that are a part of the core project. The OGX community is accepting of [new in-tree provider contributions](https://ogx-ai.github.io/docs/contributing/new_api_provider) so long as there is merit seen in the addition.

If you wish to include a new in-tree provider in the Core project, please follow these steps:

1. **Open a discussion issue**: Open an issue with label "discussion" describing the provider you want to add, its use case, and how it fits into the OGX ecosystem.

2. **Present at a community call** (optional): Bring your proposal to a [community call](README.md#community) to gather feedback, answer questions, and build consensus with other contributors and maintainers.

3. **Implement and submit a PR**: Once approved, use the [Adding a New API Provider](https://ogx-ai.github.io/docs/contributing/new_api_provider) guide to help you get started. Once approved, follow the standard pull request process outlined above.

This process helps ensure that new providers are well-designed, avoid duplication of effort, and integrate smoothly with the rest of the project.

## Repository guidelines

### Coding Style

- Comments should provide meaningful insights into the code. Avoid filler comments that simply
  describe the next step, as they create unnecessary clutter, same goes for docstrings.
- Prefer comments to clarify surprising behavior and/or relationships between parts of the code
  rather than explain what the next line of code does.
- Catching exceptions, prefer using a specific exception type rather than a broad catch-all like
  `Exception`.
- Error messages should be prefixed with "Failed to ..."
- 4 spaces for indentation rather than tab
- When using `# noqa` to suppress a style or linter warning, include a comment explaining the
  justification for bypassing the check.
- When using `# type: ignore` to suppress a mypy warning, include a comment explaining the
  justification for bypassing the check.
- Don't use unicode characters in the codebase. ASCII-only is preferred for compatibility or
  readability reasons.
- Providers configuration class should be Pydantic Field class. It should have a `description` field
  that describes the configuration. These descriptions will be used to generate the provider
  documentation.
- When possible, use keyword arguments only when calling functions.
- OGX utilizes [custom Exception classes](src/ogx_api/common/errors.py) for certain Resources that should be used where applicable.

### License

By contributing to Llama, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

## Common Tasks

Some tips about common tasks you work on while contributing to OGX:

### Installing dependencies of distributions

When installing dependencies for a distribution, you can use `ogx list-deps` to view and install the required packages.

Example:

```bash
cd work/
git clone https://github.com/ogx-ai/ogx.git
git clone https://github.com/meta-llama/llama-stack-client-python.git
cd ogx

# Show dependencies for a distribution
ogx list-deps <distro-name>

# Install dependencies
ogx list-deps <distro-name> | xargs -L1 uv pip install
```

### Updating distribution configurations

If you have made changes to a provider's configuration in any form (introducing a new config key, or
changing models, etc.), you should run `./scripts/distro_codegen.py` to re-generate various YAML
files as well as the documentation. You should not change `docs/source/.../distributions/` files
manually as they are auto-generated.

### Updating the provider documentation

If you have made changes to a provider's configuration, you should run `./scripts/provider_codegen.py`
to re-generate the documentation. You should not change `docs/source/.../providers/` files manually
as they are auto-generated.
Note that the provider "description" field will be used to generate the provider documentation.

### Building the Documentation

If you are making changes to the documentation at [https://ogx-ai.github.io/](https://ogx-ai.github.io/), you can use the following command to build the documentation and preview your changes.

```bash
# This rebuilds the documentation pages and the OpenAPI spec.
cd docs/
npm install
npm run gen-api-docs all
npm run build

# This will start a local server (usually at http://127.0.0.1:3000).
npm run serve
```

### Update API Documentation

If you modify or add new API endpoints, update the API documentation accordingly. You can do this by running the following command:

```bash
uv run ./scripts/run_openapi_generator.sh
```

The generated API schema will be available in `docs/static/`. Make sure to review the changes before committing.

## Release Process

For information about how OGX versions are released, including release schedules, testing requirements, and how to contribute to releases, see [RELEASE_PROCESS.md](./RELEASE_PROCESS.md).
