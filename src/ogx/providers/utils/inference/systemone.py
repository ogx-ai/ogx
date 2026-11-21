# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared SystemOne (judgment) support for TypeSafe-compatible backends.

The TypeSafe cloud API and local engines that serve the SystemOne wire protocol
natively (Ollama, llama.cpp server, SGLang, vLLM) all accept ``POST /v1/systemone``
with the same request and response shapes, modulo each engine's own extension or
diagnostic fields (see ``extension_fields``/``extension_fields_by_type``/
``top_level_extension_fields`` below). This module wraps the official
``typesafe_sdk`` client so every such provider shares one call path, one error
mapping, and one raw-body validation into the OGX wire model.
"""

import json
from typing import Any

import httpx2
from typesafe_sdk import (
    AsyncTypeSafeClient,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
)

from ogx.log import get_logger
from ogx.providers.utils.inference.http_client import build_network_client_kwargs
from ogx.providers.utils.inference.network_config import NetworkConfig
from ogx_api import SystemOneRequest, SystemOneResponse

logger = get_logger(name=__name__, category="inference::systemone")

# The typesafe SDK requires a non-empty API key at construction. Local engines ignore the
# Authorization header, so a fixed placeholder keeps the SDK happy without implying a secret.
LOCAL_ENGINE_API_KEY = "local"


class SystemOneProviderError(Exception):
    """An upstream SystemOne failure, carrying the upstream HTTP status code.

    Providers translate the SDK's typed errors into this so the core layer can surface the
    upstream status via the generic ``status_code`` path (core.exceptions.translation)
    without importing the SDK.
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def to_provider_error(exc: TypeSafeAPIError) -> SystemOneProviderError:
    # Surface 4xx as-is; map 5xx (or a missing status) to a bad-gateway, signalling an upstream
    # failure rather than an OGX error.
    status = exc.status
    if status is None or status >= 500:
        status = httpx2.codes.BAD_GATEWAY
    return SystemOneProviderError(status, str(exc))


async def systemone_judge(
    base_url: str,
    api_key: str,
    request: SystemOneRequest,
    *,
    network: NetworkConfig | None = None,
    extension_fields: frozenset[str] = frozenset(),
    extension_fields_by_type: dict[str, frozenset[str]] | None = None,
    top_level_extension_fields: frozenset[str] = frozenset(),
) -> SystemOneResponse:
    """Answer a SystemOne judgment request against a TypeSafe-compatible endpoint.

    ``base_url`` must be the backend root without a trailing ``/v1``: the SDK appends
    ``/v1/systemone`` itself. ``api_key`` must be a non-empty printable ASCII string; local
    engines ignore it, so pass :data:`LOCAL_ENGINE_API_KEY` for them. Both are always passed
    explicitly because the SDK otherwise falls back to the ``TYPESAFE_API_KEY`` and
    ``TYPESAFE_BASE_URL`` environment variables, which must never steer a call at a
    different backend than the one configured.

    ``request.model`` must already be the provider-specific model id (the InferenceRouter
    rewrites it before dispatch). Questions are forwarded as raw dictionaries, which the SDK
    validates. The response is reconstructed into OGX's ``SystemOneResponse`` from the raw
    response body -- rather than cast from the SDK's parsed object -- so any drift between
    the backend's schema and the OGX wire model is caught here and surfaces as a validation
    error instead of passing a wrong-shaped object through unchecked.

    ``extension_fields`` names answer fields a backend adds beyond the SystemOne schema on
    every answer regardless of its type (SGLang reports ``x_label_mass`` this way);
    ``extension_fields_by_type`` names fields added only to answers of a given ``type``
    (vLLM's ``noul`` answers carry ``probabilities``/``confidence`` that its ``choice``/
    ``score`` answers also carry, but as required SystemOne fields rather than an
    extension, so they can only be stripped conditionally on ``type``). Both are dropped
    from each answer before validation so the strict wire model still catches real drift.
    ``top_level_extension_fields`` names fields a backend adds alongside ``answers`` itself
    (vLLM reports ``id``/``object``/``created``/``diagnostics`` this way).
    """
    client_kwargs: dict[str, Any] = {}
    if network is not None:
        client_kwargs["http_client"] = httpx2.AsyncClient(**build_network_client_kwargs(network))

    # A new client (and its httpx2 transport) is created per call and closed below, so there
    # is no shared mutable client to race on or leak. The SDK closes a supplied http_client
    # on aclose.
    client = AsyncTypeSafeClient(api_key=api_key, base_url=base_url.rstrip("/"), **client_kwargs)
    try:
        payload: dict[str, Any] = request.model_dump(exclude_none=True)
        sdk_response = await client.system_one(
            state=payload["state"],
            questions=payload["questions"],
            model=request.model,
        )
        body = json.loads(sdk_response.raw_http_response.content)
        if isinstance(body, dict):
            for field in top_level_extension_fields:
                body.pop(field, None)
            if extension_fields or extension_fields_by_type:
                for answer in body.get("answers", {}).values():
                    if not isinstance(answer, dict):
                        continue
                    for field in extension_fields:
                        answer.pop(field, None)
                    if extension_fields_by_type:
                        answer_type = answer.get("type") or ""
                        for field in extension_fields_by_type.get(answer_type, frozenset()):
                            answer.pop(field, None)
        return SystemOneResponse.model_validate(body)
    except TypeSafeAPIResponseValidationError as exc:
        # The backend answered 200 with a structurally invalid body: an upstream failure,
        # not a client error.
        raise SystemOneProviderError(httpx2.codes.BAD_GATEWAY, str(exc)) from exc
    except TypeSafeAPIError as exc:
        raise to_provider_error(exc) from exc
    except TypeSafeAPIConnectionError as exc:
        # No HTTP response at all (engine down, DNS, refused connection, timeout): the
        # backend could not answer, so surface it as a bad gateway.
        raise SystemOneProviderError(httpx2.codes.BAD_GATEWAY, f"Failed to reach the SystemOne backend: {exc}") from exc
    finally:
        await client.aclose()
