# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Verification that a remote inference server is the engine an adapter expects.

Several OpenAI-compatible servers (llama.cpp, Text-Embeddings-Inference, vLLM)
can serve /v1/models on the same port, so listing models alone cannot tell them
apart. Each engine exposes a native endpoint with a recognizable response shape
(llama.cpp: GET /props, TEI: GET /info); adapters check it in initialize() so a
misconfigured base_url fails loudly at construction time instead of
mis-typing the models. The same check backs the adapters' health() methods.
"""

import httpx

from ogx_api import HealthResponse, HealthStatus

# Keys identifying a llama.cpp server in the GET /props response; older builds
# report slot counts as "n_slots"/"slots", newer ones as "total_slots".
_LLAMA_CPP_PROPS_KEYS = ("default_generation_settings", "total_slots", "n_slots", "slots")
# Key identifying a Text-Embeddings-Inference server in the GET /info response.
_TEI_INFO_KEYS = ("model_id",)

# Must stay below the 5.0s factory-probe budget in `ogx stack lets_go` so a
# stalled server surfaces a clean ServerUnreachableError instead of an empty
# asyncio timeout.
SIGNATURE_TIMEOUT = 3.0


class ServerUnreachableError(ValueError):
    """The configured server could not be reached at all (connection failure, DNS, or timeout)."""


def _root_url(base_url: str) -> str:
    """Derive the server root from an OpenAI-compatible base URL (strips /v1)."""
    return base_url.rstrip("/").removesuffix("/v1").rstrip("/")


async def verify_server_signature(
    base_url: str,
    path: str,
    signature_keys: tuple[str, ...],
    server_name: str,
    *,
    api_key: str | None = None,
    timeout: float = SIGNATURE_TIMEOUT,
) -> None:
    """Verify that the server at base_url is the expected engine type.

    GETs the engine-specific path on the server root and requires a JSON
    object containing at least one of signature_keys.

    :param base_url: The OpenAI-compatible base URL (e.g. http://host:8080/v1)
    :param path: Engine-specific endpoint on the server root (e.g. /props)
    :param signature_keys: Response keys identifying the engine
    :param server_name: Human-readable engine name for error messages
    :param api_key: Bearer token for servers fronted by authentication
    :param timeout: Request timeout in seconds
    :raises ServerUnreachableError: If the server cannot be reached at all
    :raises ValueError: If the server does not identify as the expected engine
    """
    root = _root_url(base_url)
    url = f"{root}{path}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
    except httpx.HTTPError as e:
        raise ServerUnreachableError(f"Failed to verify {root} is a {server_name} server: {e}") from e
    if response.status_code != 200:
        raise ValueError(
            f"Failed to verify {root} is a {server_name} server: GET {path} returned HTTP {response.status_code}"
        )
    try:
        body = response.json()
    except ValueError as e:
        raise ValueError(f"Failed to verify {root} is a {server_name} server: GET {path} did not return JSON") from e
    if not isinstance(body, dict) or not any(key in body for key in signature_keys):
        raise ValueError(
            f"Failed to verify {root} is a {server_name} server: GET {path} response is missing one of {signature_keys}"
        )


async def check_server_signature(
    base_url: str,
    path: str,
    signature_keys: tuple[str, ...],
    server_name: str,
    *,
    api_key: str | None = None,
    timeout: float = SIGNATURE_TIMEOUT,
) -> HealthResponse:
    """Check that the server at base_url is the expected engine type without raising.

    :return: HealthResponse with status OK, or ERROR with a diagnostic message
    """
    try:
        await verify_server_signature(base_url, path, signature_keys, server_name, api_key=api_key, timeout=timeout)
    except ValueError as e:
        return HealthResponse(status=HealthStatus.ERROR, message=str(e))
    return HealthResponse(status=HealthStatus.OK)


async def check_llama_cpp_server(base_url: str, *, api_key: str | None = None) -> HealthResponse:
    """Health check that the server at base_url is a llama.cpp llama-server."""
    return await check_server_signature(base_url, "/props", _LLAMA_CPP_PROPS_KEYS, "llama.cpp", api_key=api_key)


async def check_text_embeddings_inference_server(base_url: str, *, api_key: str | None = None) -> HealthResponse:
    """Health check that the server at base_url is a Text-Embeddings-Inference server."""
    return await check_server_signature(base_url, "/info", _TEI_INFO_KEYS, "Text-Embeddings-Inference", api_key=api_key)


async def verify_llama_cpp_server(base_url: str, *, api_key: str | None = None) -> None:
    """Verify that the server at base_url is a llama.cpp llama-server."""
    await verify_server_signature(base_url, "/props", _LLAMA_CPP_PROPS_KEYS, "llama.cpp", api_key=api_key)


async def verify_text_embeddings_inference_server(base_url: str, *, api_key: str | None = None) -> None:
    """Verify that the server at base_url is a Text-Embeddings-Inference server."""
    await verify_server_signature(base_url, "/info", _TEI_INFO_KEYS, "Text-Embeddings-Inference", api_key=api_key)


async def get_text_embeddings_inference_model_id(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout: float = SIGNATURE_TIMEOUT,
) -> str:
    """Get the model ID served by a Text-Embeddings-Inference server.

    TEI has no /v1/models endpoint; its native GET /info endpoint reports the
    single served model.

    :param base_url: The OpenAI-compatible base URL (e.g. http://host:8080/v1)
    :param api_key: Bearer token for servers fronted by authentication
    :param timeout: Request timeout in seconds
    :return: The served model ID
    :raises ServerUnreachableError: If the server cannot be reached at all
    :raises ValueError: If the server does not report a model ID
    """
    root = _root_url(base_url)
    url = f"{root}/info"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
    except httpx.HTTPError as e:
        raise ServerUnreachableError(f"Failed to query {root} for the served model: {e}") from e
    if response.status_code != 200:
        raise ValueError(f"Failed to query {root} for the served model: GET /info returned HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as e:
        raise ValueError(f"Failed to query {root} for the served model: GET /info did not return JSON") from e
    model_id = body.get("model_id") if isinstance(body, dict) else None
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"Failed to query {root} for the served model: GET /info response is missing model_id")
    return model_id
