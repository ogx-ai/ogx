# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import time
from urllib.parse import urlparse

# Initialize logging early before any loggers get created
from ogx.log import setup_logging

setup_logging()

import pytest
import requests
import yaml
from llama_stack_client import LlamaStackClient
from openai import OpenAI

from ogx.core.datatypes import QualifiedModel, RerankerModel, VectorStoresConfig
from ogx.core.library_client import OGXAsLibraryClient
from ogx.core.stack import run_config_from_dynamic_config_spec
from ogx.core.utils.config_resolution import resolve_config_or_distro
from ogx.env import get_env_or_fail

DEFAULT_PORT = 8321


def is_port_available(port: int, host: str = "localhost") -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            return True
    except OSError:
        return False


def start_ogx_server(config_name: str, *, log_stderr: bool | None = None) -> subprocess.Popen:
    """Start a ogx server with the given config."""
    if log_stderr is None:
        log_stderr = os.environ.get("OGX_TEST_LOG_STDERR", "1") == "1"

    # remove server.log if it exists
    if os.path.exists("server.log"):
        os.remove("server.log")

    cmd = f"ogx run {config_name}"
    devnull = open(os.devnull, "w")
    process = subprocess.Popen(
        shlex.split(cmd),
        stdout=devnull,  # redirect stdout to devnull to prevent deadlock
        stderr=subprocess.PIPE if log_stderr else subprocess.DEVNULL,
        text=True,
        env={**os.environ, "OGX_LOG_FILE": "server.log"},
        # Create new process group so we can kill all child processes
        preexec_fn=os.setsid,
    )
    return process


def wait_for_server_ready(base_url: str, timeout: int = 30, process: subprocess.Popen | None = None) -> bool:
    """Wait for the server to be ready by polling the health endpoint."""
    health_url = f"{base_url}/v1/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        if process and process.poll() is not None:
            print(f"Server process terminated with return code: {process.returncode}")
            if process.stderr:
                print(f"Server stderr: {process.stderr.read()}")
            else:
                print("Server stderr disabled. Check server.log for details.")
            return False

        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass

        # Print progress every 5 seconds
        elapsed = time.time() - start_time
        if int(elapsed) % 5 == 0 and elapsed > 0:
            print(f"Waiting for server at {base_url}... ({elapsed:.1f}s elapsed)")

        time.sleep(0.5)

    print(f"Server failed to respond within {timeout} seconds")
    return False


def stop_server_on_port(port: int, timeout: float = 10.0) -> None:
    """Terminate any server processes bound to the given port."""

    try:
        output = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return

    pids = {int(line) for line in output.splitlines() if line.strip()}
    if not pids:
        return

    deadline = time.time() + timeout
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in list(pids):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pids.discard(pid)

        while not is_port_available(port) and time.time() < deadline:
            time.sleep(0.1)

        if is_port_available(port):
            return

    raise RuntimeError(f"Unable to free port {port} for test server restart")


def get_provider_data():
    # TODO: this needs to be generalized so each provider can have a sample provider data just
    # like sample run config on which we can do replace_env_vars()
    keymap = {
        "TAVILY_SEARCH_API_KEY": "tavily_search_api_key",
        "BRAVE_SEARCH_API_KEY": "brave_search_api_key",
        "FIREWORKS_API_KEY": "fireworks_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
        "OPENAI_API_KEY": "openai_api_key",
        "TOGETHER_API_KEY": "together_api_key",
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "GROQ_API_KEY": "groq_api_key",
        "WOLFRAM_ALPHA_API_KEY": "wolfram_alpha_api_key",
    }
    provider_data = {}
    for key, value in keymap.items():
        if os.environ.get(key):
            provider_data[value] = os.environ[key]
    return provider_data


@pytest.fixture(scope="session")
def inference_provider_type(ogx_client):
    providers = ogx_client.providers.list()
    inference_providers = [p for p in providers if p.api == "inference"]
    assert len(inference_providers) > 0, "No inference providers found"
    return inference_providers[0].provider_type


@pytest.fixture(scope="session")
def client_with_models(
    ogx_client,
    text_model_id,
    vision_model_id,
    embedding_model_id,
    judge_model_id,
    rerank_model_id,
):
    client = ogx_client

    providers = [p for p in client.providers.list() if p.api == "inference"]
    assert len(providers) > 0, "No inference providers found"

    model_ids = {m.id for m in client.models.list()}

    if text_model_id and text_model_id not in model_ids:
        raise ValueError(f"text_model_id {text_model_id} not found")
    if vision_model_id and vision_model_id not in model_ids:
        raise ValueError(f"vision_model_id {vision_model_id} not found")
    if judge_model_id and judge_model_id not in model_ids:
        raise ValueError(f"judge_model_id {judge_model_id} not found")

    if embedding_model_id and embedding_model_id not in model_ids:
        raise ValueError(f"embedding_model_id {embedding_model_id} not found")

    if rerank_model_id and rerank_model_id not in model_ids:
        raise ValueError(f"rerank_model_id {rerank_model_id} not found")
    return client


@pytest.fixture(scope="session")
def available_shields(ogx_client):
    return [shield.identifier for shield in ogx_client.shields.list()]


@pytest.fixture(scope="session")
def model_providers(ogx_client):
    return {x.provider_id for x in ogx_client.providers.list() if x.api == "inference"}


@pytest.fixture(autouse=True)
def skip_if_no_model(request):
    model_fixtures = [
        "text_model_id",
        "vision_model_id",
        "embedding_model_id",
        "judge_model_id",
        "shield_id",
        "rerank_model_id",
    ]
    test_func = request.node.function

    actual_params = inspect.signature(test_func).parameters.keys()
    for fixture in model_fixtures:
        # Only check fixtures that are actually in the test function's signature
        if fixture in actual_params and fixture in request.fixturenames and not request.getfixturevalue(fixture):
            pytest.skip(f"{fixture} empty - skipping test")


@pytest.fixture(scope="session")
def ogx_client(request):
    # ideally, we could do this in session start given all the complex logs during initialization
    # don't clobber the test one-liner outputs. however, this also means all tests in a sub-directory
    # would be forced to use ogx_client, which is not what we want.
    print("\ninstantiating ogx_client")
    start_time = time.time()

    # Patch httpx to inject test ID for server-mode test isolation
    from ogx.testing.api_recorder import patch_httpx_for_test_id

    patch_httpx_for_test_id()

    client = instantiate_ogx_client(request.session)
    print(f"ogx_client instantiated in {time.time() - start_time:.3f}s")
    return client


def parse_vector_io_provider(config_string: str) -> str:
    # Split the string into individual key-value pairs
    pairs = config_string.split(",")
    for pair in pairs:
        # Split each pair into key and value
        key_value = pair.split("=")
        if len(key_value) == 2 and key_value[0].strip() == "vector_io":
            # Extract the provider after the last '::' if it exists
            return key_value[1].strip()
    return "inline::sentence-transformers"


def extract_model(model: str | None, default: str) -> str:
    if not model or "/" not in model:
        return default
    return model.split("/", 1)[1]


def instantiate_ogx_client(session):
    config = session.config.getoption("--stack-config")
    if not config:
        config = get_env_or_fail("OGX_CONFIG")

    if not config:
        raise ValueError("You must specify either --stack-config or OGX_CONFIG")

    # Handle server:<config_name> format or server:<config_name>:<port>
    # Also handles server:<distro>::<run_file.yaml> format
    if config.startswith("server:"):
        # Strip the "server:" prefix first
        config_part = config[7:]  # len("server:") == 7

        # Check for :: (distro::runfile format)
        if "::" in config_part:
            config_name = config_part
            port = int(os.environ.get("OGX_PORT", DEFAULT_PORT))
        else:
            # Single colon format: either <name> or <name>:<port>
            parts = config_part.split(":")
            config_name = parts[0]
            port = int(parts[1]) if len(parts) > 1 else int(os.environ.get("OGX_PORT", DEFAULT_PORT))

        base_url = f"http://localhost:{port}"

        force_restart = os.environ.get("OGX_TEST_FORCE_SERVER_RESTART") == "1"
        if force_restart:
            print(f"Forcing restart of the server on port {port}")
            stop_server_on_port(port)

        # Check if port is available
        if is_port_available(port):
            print(f"Starting ogx server with config '{config_name}' on port {port}...")

            # Start server
            server_process = start_ogx_server(config_name)

            # Wait for server to be ready
            if not wait_for_server_ready(base_url, timeout=120, process=server_process):
                print("Server failed to start within timeout")
                server_process.terminate()
                raise RuntimeError(
                    f"Server failed to start within timeout. Check that config '{config_name}' exists and is valid. "
                    f"See server.log for details."
                )

            print(f"Server is ready at {base_url}")

            # Store process for potential cleanup (pytest will handle termination at session end)
            session._ogx_server_process = server_process
        else:
            print(f"Port {port} is already in use, assuming server is already running...")

        return LlamaStackClient(
            base_url=base_url,
            provider_data=get_provider_data(),
            timeout=int(os.environ.get("OGX_CLIENT_TIMEOUT", "30")),
        )

    # check if this looks like a URL using proper URL parsing
    try:
        parsed_url = urlparse(config)
        if parsed_url.scheme and parsed_url.netloc:
            return LlamaStackClient(
                base_url=config,
                provider_data=get_provider_data(),
            )
    except Exception:
        # If URL parsing fails, treat as non-URL config
        pass

    if "=" in config:
        run_config = run_config_from_dynamic_config_spec(config)

        # --stack-config bypasses template so need this to set default embedding and reranker models
        if "vector_io" in config and "inference" in config:
            embedding_model_opt = session.config.getoption("embedding_model") or ""
            # Model identifiers are in provider_id/model_id format; extract the provider.
            provider_id = embedding_model_opt.split("/")[0] if "/" in embedding_model_opt else "sentence-transformers"
            passed_model = extract_model(session.config.getoption("embedding_model"), "nomic-ai/nomic-embed-text-v1.5")
            passed_emb = session.config.getoption("embedding_dimension")

            rerank_model_opt = session.config.getoption("rerank_model") or ""
            reranker_model = None
            if rerank_model_opt:
                provider_id_of_reranker = rerank_model_opt.split("/")[0] if "/" in rerank_model_opt else "transformers"
                passed_reranker_model = extract_model(rerank_model_opt, "Qwen/Qwen3-Reranker-0.6B")
                reranker_model = RerankerModel(
                    provider_id=provider_id_of_reranker,
                    model_id=passed_reranker_model,
                )

            run_config.vector_stores = VectorStoresConfig(
                default_embedding_model=QualifiedModel(
                    provider_id=provider_id,
                    model_id=passed_model,
                    embedding_dimensions=passed_emb,
                ),
                default_reranker_model=reranker_model,
            )

        run_config_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        with open(run_config_file.name, "w") as f:
            yaml.dump(run_config.model_dump(mode="json"), f)
        config = run_config_file.name
    elif "::" in config:
        # Handle distro::config.yaml format (e.g., ci-tests::run.yaml)
        config = str(resolve_config_or_distro(config))

    client = OGXAsLibraryClient(
        config,
        provider_data=get_provider_data(),
        skip_logger_removal=True,
    )
    return client


@pytest.fixture(scope="session")
def require_server(ogx_client):
    """
    Skip test if no server is running.

    We use the ogx_client to tell if a server was started or not.

    We use this with openai_client because it relies on a running server.
    """
    if isinstance(ogx_client, OGXAsLibraryClient):
        pytest.skip("No server running")


@pytest.fixture(scope="session")
def openai_client(ogx_client, require_server):
    base_url = f"{ogx_client.base_url}/v1"
    client = OpenAI(base_url=base_url, api_key="fake", max_retries=0, timeout=30.0)
    yield client
    # Cleanup: close HTTP connections
    try:
        client.close()
    except Exception:
        pass


@pytest.fixture(params=["openai_client", "client_with_models"])
def compat_client(request, client_with_models):
    if request.param == "openai_client" and isinstance(client_with_models, OGXAsLibraryClient):
        # OpenAI client expects a server, so unless we also rewrite OpenAI client's requests
        # to go via the Stack library client (which itself rewrites requests to be served inline),
        # we cannot do this.
        #
        # This means when we are using Stack as a library, we will test only via the OGX client.
        # When we are using a server setup, we can exercise both OpenAI and OGX clients.
        pytest.skip("(OpenAI) Compat client cannot be used with Stack library client")

    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session", autouse=True)
def cleanup_server_process(request):
    """Cleanup server process at the end of the test session."""
    yield  # Run tests

    if hasattr(request.session, "_ogx_server_process"):
        server_process = request.session._ogx_server_process
        if server_process:
            if server_process.poll() is None:
                print("Terminating ogx server process...")
            else:
                print(f"Server process already terminated with return code: {server_process.returncode}")
                return
            try:
                print(f"Terminating process {server_process.pid} and its group...")
                # Kill the entire process group
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                server_process.wait(timeout=10)
                print("Server process and children terminated gracefully")
            except subprocess.TimeoutExpired:
                print("Server process did not terminate gracefully, killing it")
                # Force kill the entire process group
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                server_process.wait()
                print("Server process and children killed")
            except Exception as e:
                print(f"Error during server cleanup: {e}")
        else:
            print("Server process not found - won't be able to cleanup")
