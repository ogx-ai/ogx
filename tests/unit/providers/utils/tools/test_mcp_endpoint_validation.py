# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.providers.utils.tools.mcp import list_mcp_tools, validate_mcp_endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1/mcp",
        "http://10.0.0.1/mcp",
        "http://192.168.1.1/mcp",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/mcp",
        "file:///etc/passwd",
        "gopher://example.com/1",
        "ftp://example.com/mcp",
    ],
)
def test_validate_mcp_endpoint_rejects_private_and_non_http_urls(endpoint: str) -> None:
    with pytest.raises(ValueError, match="Failed to"):
        validate_mcp_endpoint(endpoint)


def test_validate_mcp_endpoint_allows_public_https_url() -> None:
    with patch("ogx.providers.utils.tools.mcp.validate_url_not_private") as mock_validate:
        validate_mcp_endpoint("https://mcp.example.com/mcp")
        mock_validate.assert_called_once_with("https://mcp.example.com/mcp")


async def test_list_mcp_tools_still_allows_private_admin_endpoints() -> None:
    """Admin-configured connectors/toolgroups may use private MCP URLs.

    SSRF validation for caller-supplied Responses server_url lives in the
    Responses path, not in the shared MCP helpers.
    """
    mock_session = AsyncMock()
    mock_session.list_tools.return_value = MagicMock(tools=[])
    mock_wrapper = MagicMock()
    mock_wrapper.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_wrapper.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch("ogx.providers.utils.tools.mcp.client_wrapper", mock_wrapper):
        result = await list_mcp_tools(endpoint="http://10.0.0.5/mcp")

    assert result.data == []
    mock_wrapper.assert_called_once()
