# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import hashlib
import logging
import os
import socket
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import fire
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class McpOAuthHelper:
    """A simpler helper for OAuth2 authentication with MCP servers with OAuth discovery."""

    def __init__(self, server_url):
        self.server_url = server_url
        self.server_base_url = get_base_url(server_url)
        self.access_token = None

        # For PKCE (Proof Key for Code Exchange)
        self.code_verifier = None
        self.code_challenge = None

        # OAuth client registration
        self.client_id = None
        self.client_secret = None
        self.registered_redirect_uris = []

        # Callback server
        self.callback_port = find_available_port(8000, 8100)
        self.redirect_uri = f"http://localhost:{self.callback_port}/callback"
        self.auth_code = None
        self.auth_error = None
        self.http_server = None
        self.server_thread = None

        # Software statement for DCR
        self.software_statement = {
            "software_id": "simple-mcp-client",
            "software_version": "1.0.0",
            "software_name": "Simple MCP Client Example",
            "software_description": "A simple MCP client for demonstration purposes",
            "software_uri": "https://github.com/example/simple-mcp-client",
            "redirect_uris": [self.redirect_uri],
            "client_name": "Simple MCP Client",
            "client_uri": "https://example.com/mcp-client",
            "token_endpoint_auth_method": "none",  # Public client
        }

    def discover_auth_endpoints(self):
        """
        Discover the OAuth server metadata according to RFC8414.
        MCP servers MUST support this discovery mechanism.
        """
        well_known_url = f"{self.server_base_url}/.well-known/oauth-authorization-server"
        response = requests.get(well_known_url)
        if response.status_code == 200:
            metadata = response.json()
            logger.info("✅ Successfully discovered OAuth metadata")
            return metadata

        raise Exception(f"OAuth metadata discovery failed with status: {response.status_code}")

    def register_client(self, registration_endpoint):
        headers = {"Content-Type": "application/json"}

        registration_request = {
            "client_name": self.software_statement["client_name"],
            "redirect_uris": [self.redirect_uri],
            "token_endpoint_auth_method": "none",  # Public client
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "scope": "openid",
            "software_id": self.software_statement["software_id"],
            "software_version": self.software_statement["software_version"],
        }

        response = requests.post(registration_endpoint, headers=headers, json=registration_request)

        if response.status_code in (201, 200):
            registration_data = response.json()
            self.client_id = registration_data.get("client_id")
            self.client_secret = registration_data.get("client_secret")
            self.registered_redirect_uris = registration_data.get("redirect_uris", [self.redirect_uri])

            logger.info(f"Client ID: {self.client_id}")
            return registration_data

        raise Exception(f"Client registration failed: {response.status_code}")

    def generate_pkce_values(self):
        """Generate PKCE code verifier and challenge."""
        # Generate a random code verifier
        code_verifier = base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8").rstrip("=")

        # Generate the code challenge using SHA-256
        code_challenge_digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
        code_challenge = base64.urlsafe_b64encode(code_challenge_digest).decode("utf-8").rstrip("=")

        self.code_verifier = code_verifier
        self.code_challenge = code_challenge

        return code_verifier, code_challenge

    def stop_server(self):
        time.sleep(1)
        if self.http_server:
            self.http_server.shutdown()

    def start_callback_server(self):
        def auth_callback(auth_code: str | None, error: str | None):
            logger.info(f"Authorization callback received: auth_code={auth_code}, error={error}")
            self.auth_code = auth_code
            self.auth_error = error
            threading.Thread(target=self.stop_server).start()

        self.http_server = CallbackServer(("localhost", self.callback_port), auth_callback)

        self.server_thread = threading.Thread(target=self.http_server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        logger.info(f"🌐 Callback server started on port {self.callback_port}")

    def exchange_code_for_token(self, auth_code, token_endpoint):
        logger.info("Exchanging authorization code for access token...")

        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": self.code_verifier,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(token_endpoint, data=data, headers=headers)
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data.get("access_token")
            logger.info(f"✅ Successfully obtained access token: {self.access_token}")
            return self.access_token

        raise Exception(f"Failed to exchange code for token: {response.status_code}")

    def initiate_auth_flow(self):
        auth_metadata = self.discover_auth_endpoints()
        registration_endpoint = auth_metadata.get("registration_endpoint")
        if registration_endpoint and not self.client_id:
            self.register_client(registration_endpoint)

        self.generate_pkce_values()

        self.start_callback_server()

        auth_url = auth_metadata.get("authorization_endpoint")
        if not auth_url:
            raise Exception("No authorization endpoint in metadata")

        token_endpoint = auth_metadata.get("token_endpoint")
        if not token_endpoint:
            raise Exception("No token endpoint in metadata")

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "state": str(uuid.uuid4()),  # Random state
            "code_challenge": self.code_challenge,
            "code_challenge_method": "S256",
            "scope": "openid",  # Add appropriate scopes for Asana
        }

        full_auth_url = f"{auth_url}?{urllib.parse.urlencode(params)}"
        logger.info(f"Opening browser to authorize URL: {full_auth_url}")
        logger.info("Flow will continue after you log in")

        import webbrowser

        webbrowser.open(full_auth_url)
        self.server_thread.join(60)  # Wait up to 1 minute

        if self.auth_code:
            return self.exchange_code_for_token(self.auth_code, token_endpoint)
        elif self.auth_error:
            logger.error(f"Authorization failed: {self.auth_error}")
            return None
        else:
            logger.error("Timed out waiting for authorization")
            return None


def get_base_url(url):
    parsed_url = urllib.parse.urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"


def find_available_port(start_port, end_port):
    """Find an available port within a range."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available ports in range {start_port}-{end_port}")


class CallbackServer(HTTPServer):
    class OAuthCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed_path = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_path.query)

            if parsed_path.path == "/callback":
                auth_code = query_params.get("code", [None])[0]
                error = query_params.get("error", [None])[0]

                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()

                if error:
                    self.wfile.write(b"<html><head><title>Authorization Failed</title></head>")
                    self.wfile.write(f"<body><h1>Authorization Failed</h1><p>Error: {error}</p></body></html>".encode())
                    self.server.auth_code_callback(None, error)
                elif auth_code:
                    self.wfile.write(b"<html><head><title>Authorization Successful</title></head>")
                    self.wfile.write(
                        b"<body><h1>Authorization Successful</h1><p>You can close this window now.</p></body></html>"
                    )
                    # Call the callback with the auth code
                    self.server.auth_code_callback(auth_code, None)
                else:
                    self.wfile.write(b"<html><head><title>Authorization Failed</title></head>")
                    self.wfile.write(
                        b"<body><h1>Authorization Failed</h1><p>No authorization code received.</p></body></html>"
                    )
                    self.server.auth_code_callback(None, "No authorization code received")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self):
            """Override to suppress HTTP server logs."""
            return

    def __init__(self, server_address, auth_code_callback):
        self.auth_code_callback = auth_code_callback
        super().__init__(server_address, self.OAuthCallbackHandler)


def get_oauth_token_for_mcp_server(url: str) -> str | None:
    helper = McpOAuthHelper(url)
    return helper.initiate_auth_flow()


async def run_main(url: str):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    token = get_oauth_token_for_mcp_server(url)
    if not token:
        return

    headers = {
        "Authorization": f"Bearer {token}",
    }

    async with sse_client(url, headers=headers) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            result = await session.list_tools()

            logger.info(f"Tools: {len(result.tools)}, showing first 5:")
            for t in result.tools[:5]:
                logger.info(f"{t.name}: {t.description}")


def main(url: str):
    asyncio.run(run_main(url))


if __name__ == "__main__":
    fire.Fire(main)
