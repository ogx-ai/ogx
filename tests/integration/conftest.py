# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import inspect
import itertools
import logging  # allow-direct-logging
import os
import tempfile
import textwrap
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

from ogx.core.stack import run_config_from_dynamic_config_spec
from ogx.log import get_logger
from ogx.testing.api_recorder import patch_httpx_for_test_id

from .suites import SETUP_DEFINITIONS, SUITE_DEFINITIONS

logger = get_logger(__name__, category="tests")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        item.execution_outcome = report.outcome
        item.was_xfail = getattr(report, "wasxfail", False)


def pytest_sessionstart(session):
    # stop macOS from complaining about duplicate OpenMP libraries
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if "OGX_TEST_INFERENCE_MODE" not in os.environ:
        os.environ["OGX_TEST_INFERENCE_MODE"] = "replay"

    if "OGX_LOGGING" not in os.environ:
        os.environ["OGX_LOGGING"] = "all=warning"

    if "SQLITE_STORE_DIR" not in os.environ:
        os.environ["SQLITE_STORE_DIR"] = tempfile.mkdtemp()
        logger.info(f"Setting SQLITE_STORE_DIR: {os.environ['SQLITE_STORE_DIR']}")

    # Set test stack config type for api_recorder test isolation
    stack_config = session.config.getoption("--stack-config", default=None)
    if stack_config and (
        stack_config.startswith("server:") or stack_config.startswith("docker:") or stack_config.startswith("http")
    ):
        os.environ["OGX_TEST_STACK_CONFIG_TYPE"] = "server"
        logger.info(f"Test stack config type: server (stack_config={stack_config})")
    else:
        os.environ["OGX_TEST_STACK_CONFIG_TYPE"] = "library_client"
        logger.info(f"Test stack config type: library_client (stack_config={stack_config})")

    patch_httpx_for_test_id()


@pytest.fixture(autouse=True)
def suppress_httpx_logs(caplog):
    """Suppress httpx INFO logs for all integration tests"""
    caplog.set_level(logging.WARNING, logger="httpx")


@pytest.fixture(autouse=True)
def _track_test_context(request):
    """Automatically track current test context for isolated recordings.

    This fixture runs for every test and stores the test's nodeid in a contextvar
    that the recording system can access to determine which subdirectory to use.
    """
    from ogx.core.testing_context import reset_test_context, set_test_context

    token = set_test_context(request.node.nodeid)

    yield

    reset_test_context(token)


def pytest_runtest_teardown(item):
    # Check if the test actually ran and passed or failed, but was not skipped or an expected failure (xfail)
    outcome = getattr(item, "execution_outcome", None)
    was_xfail = getattr(item, "was_xfail", False)

    name = item.nodeid
    if not any(x in name for x in ("inference/", "safety/", "agents/")):
        return

    logger.debug(f"Test '{item.nodeid}' outcome was '{outcome}' (xfail={was_xfail})")
    if outcome in ("passed", "failed") and not was_xfail:
        interval_seconds = os.getenv("OGX_TEST_INTERVAL_SECONDS")
        if interval_seconds:
            time.sleep(float(interval_seconds))


def pytest_configure(config):
    config.option.tbstyle = "short"
    config.option.disable_warnings = True

    load_dotenv()

    env_vars = config.getoption("--env") or []
    for env_var in env_vars:
        key, value = env_var.split("=", 1)
        os.environ[key] = value

    inference_mode = config.getoption("--inference-mode")
    os.environ["OGX_TEST_INFERENCE_MODE"] = inference_mode

    suite = config.getoption("--suite")
    if suite:
        if suite not in SUITE_DEFINITIONS:
            raise pytest.UsageError(f"Unknown suite: {suite}. Available: {', '.join(sorted(SUITE_DEFINITIONS.keys()))}")

    # Apply setups (global parameterizations): env + defaults
    setup = config.getoption("--setup")
    if suite and not setup:
        setup = SUITE_DEFINITIONS[suite].default_setup

    if setup:
        if setup not in SETUP_DEFINITIONS:
            raise pytest.UsageError(
                f"Unknown setup '{setup}'. Available: {', '.join(sorted(SETUP_DEFINITIONS.keys()))}"
            )

        setup_obj = SETUP_DEFINITIONS[setup]
        logger.info(f"Applying setup '{setup}'{' for suite ' + suite if suite else ''}")
        # Apply env first
        for k, v in setup_obj.env.items():
            if k not in os.environ:
                os.environ[k] = str(v)
        # Apply defaults if not provided explicitly
        for dest, value in setup_obj.defaults.items():
            current = getattr(config.option, dest, None)
            if current is None:
                setattr(config.option, dest, value)

    # Apply global fallback for embedding_dimension if still not set
    if getattr(config.option, "embedding_dimension", None) is None:
        config.option.embedding_dimension = 384

    # Apply global fallback for embedding_model when using stack configs with embedding models
    if getattr(config.option, "embedding_model", None) is None:
        stack_config = config.getoption("--stack-config", default=None)
        if stack_config and "=" in stack_config:
            run_config = run_config_from_dynamic_config_spec(stack_config)
            inference_providers = run_config.providers.get("inference", [])
            if any("sentence-transformers" in p.provider_type for p in inference_providers):
                config.option.embedding_model = "sentence-transformers/nomic-ai/nomic-embed-text-v1.5"


def pytest_addoption(parser):
    parser.addoption(
        "--stack-config",
        help=textwrap.dedent(
            """
            a 'pointer' to the stack. this can be either be:
            (a) a template name like `starter`, or
            (b) a path to a config.yaml file, or
            (c) a dynamic config spec, e.g. `inference=fireworks,safety=llama-guard,agents=builtin`, or
            (d) a server config like `server:ci-tests`, or
            (e) a docker config like `docker:ci-tests` (builds and runs container)
            """
        ),
    )
    parser.addoption("--env", action="append", help="Set environment variables, e.g. --env KEY=value")
    parser.addoption(
        "--text-model",
        help="comma-separated list of text models. Fixture name: text_model_id",
    )
    parser.addoption(
        "--vision-model",
        help="comma-separated list of vision models. Fixture name: vision_model_id",
    )
    parser.addoption(
        "--embedding-model",
        help="comma-separated list of embedding models. Fixture name: embedding_model_id",
    )
    parser.addoption(
        "--rerank-model",
        help="comma-separated list of rerank models. Fixture name: rerank_model_id",
    )
    parser.addoption(
        "--safety-shield",
        help="comma-separated list of safety shields. Fixture name: shield_id",
    )
    parser.addoption(
        "--judge-model",
        help="Specify the judge model to use for testing",
    )
    parser.addoption(
        "--embedding-dimension",
        type=int,
        default=768,
        help="Output dimensionality of the embedding model to use for testing. Default: 768",
    )

    parser.addoption(
        "--inference-mode",
        help="Inference mode: { record, replay, live, record-if-missing } (default: replay)",
        choices=["record", "replay", "live", "record-if-missing"],
        default="replay",
    )
    parser.addoption(
        "--report",
        help="Path where the test report should be written, e.g. --report=/path/to/report.md",
    )

    available_suites = ", ".join(sorted(SUITE_DEFINITIONS.keys()))
    suite_help = (
        f"Single test suite to run (narrows collection). Available: {available_suites}. Example: --suite=responses"
    )
    parser.addoption("--suite", help=suite_help)

    # Global setups for any suite
    available_setups = ", ".join(sorted(SETUP_DEFINITIONS.keys()))
    setup_help = (
        f"Global test setup configuration. Available: {available_setups}. "
        "Can be used with any suite. Example: --setup=ollama"
    )
    parser.addoption("--setup", help=setup_help)


MODEL_SHORT_IDS = {
    "meta-llama/Llama-3.2-3B-Instruct": "3B",
    "meta-llama/Llama-3.1-8B-Instruct": "8B",
    "meta-llama/Llama-3.1-70B-Instruct": "70B",
    "meta-llama/Llama-3.1-405B-Instruct": "405B",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "11B",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "90B",
    "meta-llama/Llama-3.3-70B-Instruct": "70B",
    "meta-llama/Llama-Guard-3-1B": "Guard1B",
    "meta-llama/Llama-Guard-3-8B": "Guard8B",
    "nomic-ai/nomic-embed-text-v1.5": "Nomic-v1.5",
}


def get_short_id(value):
    return MODEL_SHORT_IDS.get(value, value)


def pytest_generate_tests(metafunc):
    """
    This is the main function which processes CLI arguments and generates various combinations of parameters.
    It is also responsible for generating test IDs which are succinct enough.

    Each option can be comma separated list of values which results in multiple parameter combinations.
    """
    # Handle vector_io_provider_id dynamically
    if "vector_io_provider_id" in metafunc.fixturenames:
        config_str = metafunc.config.getoption("--stack-config", default=None) or os.environ.get("OGX_CONFIG")
        providers = None
        if config_str and "=" in config_str:
            run_config = run_config_from_dynamic_config_spec(config_str)
            providers = [p.provider_id for p in run_config.providers.get("vector_io", [])]
        if providers is None:
            inference_mode = os.environ.get("OGX_TEST_INFERENCE_MODE")
            providers = (
                ["faiss", "sqlite-vec", "milvus", "chromadb", "pgvector", "weaviate", "qdrant"]
                if inference_mode == "live"
                else ["faiss", "sqlite-vec"]
            )
        metafunc.parametrize("vector_io_provider_id", providers, ids=[f"vector_io={p}" for p in providers])

    params = []
    param_values = {}
    id_parts = []

    # Map of fixture name to its CLI option and ID prefix
    fixture_configs = {
        "text_model_id": ("--text-model", "txt"),
        "vision_model_id": ("--vision-model", "vis"),
        "embedding_model_id": ("--embedding-model", "emb"),
        "shield_id": ("--safety-shield", "shield"),
        "judge_model_id": ("--judge-model", "judge"),
        "embedding_dimension": ("--embedding-dimension", "dim"),
        "rerank_model_id": ("--rerank-model", "rerank"),
    }

    # Collect all parameters and their values
    for fixture_name, (option, id_prefix) in fixture_configs.items():
        if fixture_name not in metafunc.fixturenames:
            continue

        params.append(fixture_name)
        # Use getattr on config.option to see values set by pytest_configure fallbacks
        dest = option.lstrip("-").replace("-", "_")
        val = getattr(metafunc.config.option, dest, None)

        values = [v.strip() for v in str(val).split(",")] if val else [None]
        param_values[fixture_name] = values
        if val:
            id_parts.extend(f"{id_prefix}={get_short_id(v)}" for v in values)

    if not params:
        return

    # Generate all combinations of parameter values
    value_combinations = list(itertools.product(*[param_values[p] for p in params]))

    # Generate test IDs
    test_ids = []
    non_empty_params = [(i, values) for i, values in enumerate(param_values.values()) if values[0] is not None]

    # Get actual function parameters using inspect
    test_func_params = set(inspect.signature(metafunc.function).parameters.keys())

    if non_empty_params:
        # For each combination, build an ID from the non-None parameters
        for combo in value_combinations:
            parts = []
            for param_name, val in zip(params, combo, strict=True):
                # Only include if parameter is in test function signature and value is meaningful
                if param_name in test_func_params and val:
                    prefix = fixture_configs[param_name][1]  # Get the ID prefix
                    parts.append(f"{prefix}={get_short_id(val)}")
            if parts:
                test_ids.append(":".join(parts))

    metafunc.parametrize(params, value_combinations, scope="session", ids=test_ids if test_ids else None)


def pytest_ignore_collect(path: str, config: pytest.Config) -> bool:
    """Skip collecting paths outside the selected suite roots for speed."""
    suite = config.getoption("--suite")
    if not suite:
        return False

    sobj = SUITE_DEFINITIONS.get(suite)
    roots: list[str] = sobj.get("roots", []) if isinstance(sobj, dict) else getattr(sobj, "roots", [])
    if not roots:
        return False

    p = Path(str(path)).resolve()

    # Only constrain within tests/integration to avoid ignoring unrelated tests
    integration_root = (Path(str(config.rootpath)) / "tests" / "integration").resolve()
    if not p.is_relative_to(integration_root):
        return False

    for r in roots:
        # Handle pytest node IDs like "path/to/file.py::test_function"
        file_path = r.split("::")[0] if "::" in r else r
        rp = (Path(str(config.rootpath)) / file_path).resolve()
        if rp.is_file():
            # Allow the exact file and any ancestor directories so pytest can walk into it.
            if p == rp:
                return False
            if p.is_dir() and rp.is_relative_to(p):
                return False
        else:
            # Allow anything inside an allowed directory
            if p.is_relative_to(rp):
                return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Filter collected tests to only those matching suite roots with ::test_function specifiers."""
    suite = config.getoption("--suite")
    if not suite:
        return

    sobj = SUITE_DEFINITIONS.get(suite)
    roots: list[str] = sobj.get("roots", []) if isinstance(sobj, dict) else getattr(sobj, "roots", [])
    if not roots:
        return

    # Check if any roots have ::test_function specifiers
    test_specifiers = [r for r in roots if "::" in r]
    if not test_specifiers:
        return  # No filtering needed for directory/file-only roots

    # Build set of allowed (file, test_function) tuples
    allowed_tests: set[tuple[str, str]] = set()
    for r in test_specifiers:
        file_path, test_name = r.split("::", 1)
        allowed_tests.add((file_path, test_name))

    allowed_roots: list[Path] = []
    for r in roots:
        if "::" not in r:
            allowed_roots.append((config.rootpath / r).resolve())

    # Filter items to only those matching the allowed tests
    selected = []
    for item in items:
        # Get the file path relative to rootdir
        rel_path = str(Path(item.fspath).relative_to(config.rootpath))
        # Get the test function name (without parametrization)
        test_name = item.originalname if hasattr(item, "originalname") else item.name.split("[")[0]

        if (rel_path, test_name) in allowed_tests:
            selected.append(item)
            continue

        item_path = Path(item.fspath).resolve()
        for root in allowed_roots:
            if root.is_file() and item_path == root:
                selected.append(item)
                break
            elif root.is_dir() and item_path.is_relative_to(root):
                selected.append(item)
                break

    items[:] = selected


def get_vector_io_provider_ids(client):
    """Get all available vector_io provider IDs."""
    providers = [p for p in client.providers.list() if p.api == "vector_io"]
    return [p.provider_id for p in providers]


def vector_provider_wrapper(func):
    """Decorator with runtime validation and fallback parametrization."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import inspect

        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        vector_io_provider_id = bound_args.arguments.get("vector_io_provider_id")
        if not vector_io_provider_id:
            pytest.skip("No vector_io_provider_id provided")

        # Get client_with_models to check available providers
        client_with_models = bound_args.arguments.get("client_with_models")
        if client_with_models:
            available_providers = get_vector_io_provider_ids(client_with_models)
            if vector_io_provider_id not in available_providers:
                pytest.skip(f"Provider '{vector_io_provider_id}' not available. Available: {available_providers}")

        return func(*args, **kwargs)

    # Always return just the wrapper - pytest_generate_tests handles parametrization
    # If pytest_generate_tests doesn't parametrize, that means there was no
    # vector_io_provider_id in fixturenames, so no parametrization is needed
    return wrapper


@pytest.fixture
def vector_io_provider_id(request, client_with_models):
    """Fixture that provides a specific vector_io provider ID, skipping if not available."""
    if hasattr(request, "param"):
        requested_provider = request.param
        available_providers = get_vector_io_provider_ids(client_with_models)

        if requested_provider not in available_providers:
            pytest.skip(f"Provider '{requested_provider}' not available. Available: {available_providers}")

        return requested_provider
    else:
        provider_ids = get_vector_io_provider_ids(client_with_models)
        if not provider_ids:
            pytest.skip("No vector_io providers available")
        return provider_ids[0]
