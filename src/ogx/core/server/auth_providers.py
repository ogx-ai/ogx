# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import ssl
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
import jwt
from pydantic import BaseModel, Field
from starlette.types import Scope

from ogx.core.datatypes import (
    AuthenticationConfig,
    CustomAuthConfig,
    GitHubTokenAuthConfig,
    KubernetesAuthProviderConfig,
    OAuth2TokenAuthConfig,
    UpstreamHeaderAuthConfig,
    User,
)
from ogx.log import get_logger
from ogx_api import TokenValidationError

logger = get_logger(name=__name__, category="core::auth")

# FIPS-approved JWT signing algorithms (asymmetric only).
# Symmetric algorithms (HS256, etc.) are excluded to prevent algorithm confusion attacks.
FIPS_APPROVED_JWT_ALGORITHMS = [
    "RS256",
    "RS384",
    "RS512",
    "PS256",
    "PS384",
    "PS512",
    "ES256",
    "ES384",
    "ES512",
]


class AuthResponse(BaseModel):
    """The format of the authentication response from the auth endpoint."""

    principal: str
    # further attributes that may be used for access control decisions
    attributes: dict[str, list[str]] | None = None
    message: str | None = Field(
        default=None, description="Optional message providing additional context about the authentication result."
    )


class AuthRequestContext(BaseModel):
    """Context information about the HTTP request being authenticated."""

    path: str = Field(description="The path of the request being authenticated")

    headers: dict[str, str] = Field(description="HTTP headers from the original request (excluding Authorization)")

    params: dict[str, list[str]] = Field(default_factory=dict, description="Query parameters from the original request")


class AuthRequest(BaseModel):
    """Request payload sent to custom authentication endpoints."""

    api_key: str = Field(description="The API key extracted from the Authorization header")

    request: AuthRequestContext = Field(description="Context information about the request being authenticated")


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""

    @property
    def requires_http_bearer(self) -> bool:
        """Whether this provider requires a Bearer token from the Authorization header.

        Providers that extract identity from other sources (e.g. gateway-injected
        headers) should override this to return False.
        """
        return True

    @abstractmethod
    async def validate_token(self, token: str, scope: Scope | None = None) -> User:
        """Validate a token and return access attributes."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up any resources."""
        pass

    def get_auth_error_message(self, scope: Scope | None = None) -> str:
        """Return provider-specific authentication error message."""
        return "Authentication required"


def get_attributes_from_claims(claims: dict[str, Any], mapping: dict[str, str]) -> dict[str, list[str]]:
    """Extract user attributes from token claims using the configured claims-to-attributes mapping.

    Args:
        claims: Token claims dictionary (e.g., from JWT or introspection).
        mapping: Dictionary mapping claim keys to attribute keys.

    Returns:
        Dictionary mapping attribute keys to lists of attribute values.
    """
    attributes: dict[str, list[str]] = {}
    for claim_key, attribute_key in mapping.items():
        # First try dot notation for nested traversal (e.g., "resource_access.ogx.roles")
        # Then fall back to literal key with dots (e.g., "my.dotted.key")
        claim: object = claims
        keys = claim_key.split(".")
        for key in keys:
            if isinstance(claim, dict) and key in claim:
                claim = claim[key]
            else:
                claim = None
                break

        if claim is None and claim_key in claims:
            # Fall back to checking if claim_key exists as a literal key
            claim = claims[claim_key]

        if claim is None:
            continue

        if isinstance(claim, list):
            values = claim
        elif isinstance(claim, str):
            values = claim.split()
        else:
            continue

        if attribute_key in attributes:
            attributes[attribute_key].extend(values)
        else:
            attributes[attribute_key] = values
    return attributes


class OAuth2TokenAuthProvider(AuthProvider):
    """
    JWT token authentication provider that validates a JWT token and extracts access attributes.

    This should be the standard authentication provider for most use cases.
    """

    def __init__(self, config: OAuth2TokenAuthConfig) -> None:
        self.config = config
        self._jwks_client: jwt.PyJWKClient | None = None

    async def validate_token(self, token: str, scope: Scope | None = None) -> User:
        if self.config.jwks:
            return await self.validate_jwt_token(token, scope)
        if self.config.introspection:
            return await self.introspect_token(token, scope)
        raise ValueError("One of jwks or introspection must be configured")

    def _get_jwks_client(self) -> jwt.PyJWKClient:
        if self._jwks_client is None:
            ssl_context = None
            if not self.config.verify_tls:
                # Disable SSL verification if verify_tls is False
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            elif self.config.tls_cafile:
                # Use custom CA file if provided
                ssl_context = ssl.create_default_context(
                    cafile=self.config.tls_cafile.as_posix(),
                )
                # If verify_tls is True and no tls_cafile, ssl_context remains None (use system defaults)

            # Prepare headers for JWKS request - this is needed for Kubernetes to authenticate
            # to the JWK endpoint, we must use the token in the config to authenticate
            headers = {}
            if self.config.jwks and self.config.jwks.token:
                headers["Authorization"] = f"Bearer {self.config.jwks.token}"

            # Ensure uri is not None for PyJWKClient
            if not self.config.jwks or not self.config.jwks.uri:
                raise ValueError("JWKS configuration requires a valid URI")

            # Build kwargs conditionally to avoid passing None values
            jwks_kwargs: dict[str, Any] = {
                "cache_keys": True,
                "max_cached_keys": 10,
                "headers": headers,
                "ssl_context": ssl_context,
            }
            if self.config.jwks.key_recheck_period is not None:
                jwks_kwargs["lifespan"] = self.config.jwks.key_recheck_period

            self._jwks_client = jwt.PyJWKClient(self.config.jwks.uri, **jwks_kwargs)
        return self._jwks_client

    async def validate_jwt_token(self, token: str, scope: Scope | None = None) -> User:
        """Validate a token using the JWT token."""
        try:
            jwks_client: jwt.PyJWKClient = self._get_jwks_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            # Decode and verify the JWT using a static allowlist of FIPS-approved algorithms.
            # Never trust the algorithm from the unverified token header (algorithm confusion attack).
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=FIPS_APPROVED_JWT_ALGORITHMS,
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={"verify_exp": True, "verify_aud": True, "verify_iss": True},
            )
        except Exception as exc:
            raise ValueError("Invalid JWT token") from exc

        # There are other standard claims, the most relevant of which is `scope`.
        # We should incorporate these into the access attributes.
        principal = claims["sub"]
        access_attributes = get_attributes_from_claims(claims, self.config.claims_mapping)
        return User(
            principal=principal,
            attributes=access_attributes,
        )

    async def introspect_token(self, token: str, scope: Scope | None = None) -> User:
        """Validate a token using token introspection as defined by RFC 7662."""
        form = {
            "token": token,
        }
        if self.config.introspection is None:
            raise ValueError("Introspection is not configured")

        # Default to True (system CA verification); only disable when verify_tls=False
        ssl_ctxt: ssl.SSLContext | bool = True
        if not self.config.verify_tls:
            ssl_ctxt = False
        elif self.config.tls_cafile:
            ssl_ctxt = ssl.create_default_context(cafile=self.config.tls_cafile.as_posix())

        # Build post kwargs conditionally based on auth method
        post_kwargs: dict[str, Any] = {
            "url": self.config.introspection.url,
            "data": form,
            "timeout": 10.0,
        }

        if self.config.introspection.send_secret_in_body:
            form["client_id"] = self.config.introspection.client_id
            form["client_secret"] = self.config.introspection.client_secret
        else:
            # httpx auth parameter expects tuple[str | bytes, str | bytes]
            post_kwargs["auth"] = (
                self.config.introspection.client_id,
                self.config.introspection.client_secret,
            )

        try:
            async with httpx.AsyncClient(verify=ssl_ctxt) as client:
                response = await client.post(**post_kwargs)
                if response.status_code != httpx.codes.OK:
                    logger.warning("Token introspection failed with status code", status_code=response.status_code)
                    raise ValueError(f"Token introspection failed: {response.status_code}")

                fields = response.json()
                if not fields["active"]:
                    raise ValueError("Token not active")
                principal = fields["sub"] or fields["username"]
                access_attributes = get_attributes_from_claims(fields, self.config.claims_mapping)
                return User(
                    principal=principal,
                    attributes=access_attributes,
                )
        except httpx.TimeoutException:
            logger.exception("Token introspection request timed out")
            raise
        except ValueError:
            # Re-raise ValueError exceptions to preserve their message
            raise
        except Exception as e:
            logger.exception("Error during token introspection")
            raise ValueError("Token introspection error") from e

    async def close(self) -> None:
        pass

    def get_auth_error_message(self, scope: Scope | None = None) -> str:
        """Return OAuth2-specific authentication error message."""
        if self.config.issuer:
            return f"Authentication required. Please provide a valid OAuth2 Bearer token from {self.config.issuer}"
        elif self.config.introspection:
            # Extract domain from introspection URL for a cleaner message
            domain = urlparse(self.config.introspection.url).netloc
            return f"Authentication required. Please provide a valid OAuth2 Bearer token validated by {domain}"
        else:
            return "Authentication required. Please provide a valid OAuth2 Bearer token in the Authorization header"


class CustomAuthProvider(AuthProvider):
    """Custom authentication provider that uses an external endpoint."""

    def __init__(self, config: CustomAuthConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def validate_token(self, token: str, scope: Scope | None = None) -> User:
        """Validate a token using the custom authentication endpoint."""
        if scope is None:
            scope = {}

        headers = dict(scope.get("headers", []))
        path = scope.get("path", "")
        request_headers = {k.decode(): v.decode() for k, v in headers.items()}

        # Remove sensitive headers
        if "authorization" in request_headers:
            del request_headers["authorization"]

        query_string = scope.get("query_string", b"").decode()
        params = parse_qs(query_string)

        # Build the auth request model
        auth_request = AuthRequest(
            api_key=token,
            request=AuthRequestContext(
                path=path,
                headers=request_headers,
                params=params,
            ),
        )

        # Validate with authentication endpoint
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.endpoint,
                    json=auth_request.model_dump(),
                    timeout=10.0,  # Add a reasonable timeout
                )
                if response.status_code != httpx.codes.OK:
                    logger.warning("Authentication failed with status code", status_code=response.status_code)
                    raise ValueError(f"Authentication failed: {response.status_code}")

                # Parse and validate the auth response
                try:
                    response_data = response.json()
                    auth_response = AuthResponse(**response_data)
                    return User(principal=auth_response.principal, attributes=auth_response.attributes)
                except Exception as e:
                    logger.exception("Error parsing authentication response")
                    raise ValueError("Invalid authentication response format") from e

        except httpx.TimeoutException:
            logger.exception("Authentication request timed out")
            raise
        except ValueError:
            # Re-raise ValueError exceptions to preserve their message
            raise
        except Exception as e:
            logger.exception("Error during authentication")
            raise ValueError("Authentication service error") from e

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def get_auth_error_message(self, scope: Scope | None = None) -> str:
        """Return custom auth provider-specific authentication error message."""
        domain = urlparse(self.config.endpoint).netloc
        if domain:
            return f"Authentication required. Please provide your API key as a Bearer token (validated by {domain})"
        else:
            return "Authentication required. Please provide your API key as a Bearer token in the Authorization header"


class GitHubTokenAuthProvider(AuthProvider):
    """
    GitHub token authentication provider that validates GitHub access tokens directly.

    This provider accepts GitHub personal access tokens or OAuth tokens and verifies
    them against the GitHub API to get user information.
    """

    def __init__(self, config: GitHubTokenAuthConfig) -> None:
        self.config = config

    async def validate_token(self, token: str, scope: Scope | None = None) -> User:
        """Validate a GitHub token by calling the GitHub API.

        This validates tokens issued by GitHub (personal access tokens or OAuth tokens).
        """
        try:
            user_info = await _get_github_user_info(token, self.config.github_api_base_url)
        except httpx.HTTPStatusError as e:
            logger.warning("GitHub token validation failed", error=str(e))
            raise ValueError("GitHub token validation failed. Please check your token and try again.") from e

        principal = user_info["user"]["login"]

        github_data = {
            "login": user_info["user"]["login"],
            "id": str(user_info["user"]["id"]),
            "organizations": user_info.get("organizations", []),
        }

        access_attributes = get_attributes_from_claims(github_data, self.config.claims_mapping)

        return User(
            principal=principal,
            attributes=access_attributes,
        )

    async def close(self) -> None:
        """Clean up any resources."""
        pass

    def get_auth_error_message(self, scope: Scope | None = None) -> str:
        """Return GitHub-specific authentication error message."""
        return "Authentication required. Please provide a valid GitHub access token (https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) in the Authorization header (Bearer <token>)"


async def _get_github_user_info(access_token: str, github_api_base_url: str) -> dict[str, Any]:
    """Fetch user info and organizations from GitHub API."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "ogx",
    }

    async with httpx.AsyncClient() as client:
        user_response = await client.get(f"{github_api_base_url}/user", headers=headers, timeout=10.0)
        user_response.raise_for_status()
        user_data = user_response.json()

        return {
            "user": user_data,
        }


class KubernetesAuthProvider(AuthProvider):
    """
    Kubernetes authentication provider that validates tokens using the Kubernetes SelfSubjectReview API.
    This provider integrates with Kubernetes API server by using the
    /apis/authentication.k8s.io/v1/selfsubjectreviews endpoint to validate tokens and extract user information.
    """

    def __init__(self, config: KubernetesAuthProviderConfig) -> None:
        self.config = config

    def _httpx_verify_value(self) -> bool | str:
        """
        Build the value for httpx's `verify` parameter.
        - False disables verification.
        - Path string points to a CA bundle.
        - True uses system defaults.
        """
        if not self.config.verify_tls:
            return False
        if self.config.tls_cafile:
            return self.config.tls_cafile.as_posix()
        return True

    async def validate_token(self, token: str, scope: Scope | None = None) -> User:
        """Validate a token using Kubernetes SelfSubjectReview API endpoint."""
        # Build the Kubernetes SelfSubjectReview API endpoint URL
        review_api_url = urljoin(self.config.api_server_url, "/apis/authentication.k8s.io/v1/selfsubjectreviews")

        # Create SelfSubjectReview request body
        review_request = {"apiVersion": "authentication.k8s.io/v1", "kind": "SelfSubjectReview"}
        verify = self._httpx_verify_value()

        try:
            async with httpx.AsyncClient(verify=verify, timeout=10.0) as client:
                response = await client.post(
                    review_api_url,
                    json=review_request,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )

                if response.status_code == httpx.codes.UNAUTHORIZED:
                    raise TokenValidationError("Invalid token")
                if response.status_code != httpx.codes.CREATED:
                    logger.warning(
                        "Kubernetes SelfSubjectReview API failed with status code", status_code=response.status_code
                    )
                    raise TokenValidationError(f"Token validation failed: {response.status_code}")

                review_response = response.json()
                # Extract user information from SelfSubjectReview response
                status = review_response.get("status", {})
                if not status:
                    raise ValueError("No status found in SelfSubjectReview response")

                user_info = status.get("userInfo", {})
                if not user_info:
                    raise ValueError("No userInfo found in SelfSubjectReview response")

                username = user_info.get("username")
                if not username:
                    raise ValueError("No username found in SelfSubjectReview response")

                # Build user attributes from Kubernetes user info
                user_attributes = get_attributes_from_claims(user_info, self.config.claims_mapping)

                return User(
                    principal=username,
                    attributes=user_attributes,
                )

        except httpx.TimeoutException:
            logger.warning("Kubernetes SelfSubjectReview API request timed out")
            raise ValueError("Token validation timeout") from None
        except Exception as e:
            logger.warning("Error during token validation", error=str(e))
            raise ValueError(f"Token validation error: {str(e)}") from e

    async def close(self) -> None:
        """Close any resources."""
        pass


class UpstreamHeaderAuthProvider(AuthProvider):
    """Authentication provider that extracts identity from upstream gateway headers.

    Used when an upstream gateway (Authorino, Istio, or any reverse proxy) handles
    authentication and injects user identity into request headers. This provider
    trusts the headers and performs no token validation or outbound calls.
    """

    def __init__(self, config: UpstreamHeaderAuthConfig) -> None:
        self.config = config

    @property
    def requires_http_bearer(self) -> bool:
        return False

    async def validate_token(self, token: str, scope: Scope | None = None) -> User:
        if scope is None:
            raise ValueError("Missing required authentication header: " + self.config.principal_header)

        headers = dict(scope.get("headers", []))

        # HTTP headers are case-insensitive; ASGI stores them as lowercase bytes
        principal_key = self.config.principal_header.lower().encode()
        principal_value = headers.get(principal_key)

        if not principal_value:
            raise ValueError("Missing required authentication header: " + self.config.principal_header)

        principal = principal_value.decode()

        attributes: dict[str, list[str]] | None = None
        if self.config.attributes_header:
            attributes_key = self.config.attributes_header.lower().encode()
            attributes_value = headers.get(attributes_key)
            if attributes_value:
                import json

                try:
                    parsed = json.loads(attributes_value.decode())
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    raise ValueError("Failed to parse authentication attributes header: invalid JSON") from e

                if not isinstance(parsed, dict):
                    raise ValueError("Failed to parse authentication attributes header: expected JSON object")

                # Normalize values to list[str] to match the User.attributes type
                attributes = {}
                for k, v in parsed.items():
                    if isinstance(v, list):
                        attributes[k] = [str(item) for item in v]
                    elif isinstance(v, str):
                        attributes[k] = [v]
                    else:
                        attributes[k] = [str(v)]

        return User(principal=principal, attributes=attributes)

    async def close(self) -> None:
        pass

    def get_auth_error_message(self, scope: Scope | None = None) -> str:
        return f"Authentication required. Upstream gateway must set the {self.config.principal_header} header"


def create_auth_provider(config: AuthenticationConfig) -> AuthProvider:
    """Factory function to create the appropriate auth provider."""
    provider_config = config.provider_config

    if isinstance(provider_config, CustomAuthConfig):
        return CustomAuthProvider(provider_config)
    elif isinstance(provider_config, OAuth2TokenAuthConfig):
        return OAuth2TokenAuthProvider(provider_config)
    elif isinstance(provider_config, GitHubTokenAuthConfig):
        return GitHubTokenAuthProvider(provider_config)
    elif isinstance(provider_config, KubernetesAuthProviderConfig):
        return KubernetesAuthProvider(provider_config)
    elif isinstance(provider_config, UpstreamHeaderAuthConfig):
        return UpstreamHeaderAuthProvider(provider_config)
    else:
        raise ValueError(f"Unknown authentication provider config type: {type(provider_config)}")
