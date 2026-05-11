# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

import pytest

from ogx.providers.utils.common.url_validation import validate_url_not_private


class TestValidateUrlNotPrivate:
    def test_rejects_localhost_ip(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://127.0.0.1/image.png")

    def test_rejects_loopback_range(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://127.0.0.2:8080/path")

    def test_rejects_10_network(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://10.0.0.1/file")

    def test_rejects_172_16_network(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://172.16.0.1/file")

    def test_rejects_172_31_network(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://172.31.255.255/file")

    def test_rejects_192_168_network(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://192.168.1.1/file")

    def test_rejects_link_local(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://169.254.0.1/file")

    def test_rejects_ipv6_loopback(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://[::1]/file")

    def test_rejects_ipv6_unique_local(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://[fc00::1]/file")

    def test_rejects_ipv6_link_local(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://[fe80::1]/file")

    def test_rejects_ipv4_mapped_ipv6_loopback(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://[::ffff:127.0.0.1]/file")

    def test_rejects_unspecified_ip(self):
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://0.0.0.0/file")

    def test_allows_public_ip(self):
        validate_url_not_private("http://8.8.8.8/file")

    def test_allows_172_outside_private_range(self):
        validate_url_not_private("http://172.32.0.1/file")

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    def test_rejects_hostname_resolving_to_private(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("192.168.1.1", 0)),
        ]
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://evil.example.com/file")

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    def test_allows_hostname_resolving_to_public(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]
        validate_url_not_private("http://example.com/file")

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    def test_rejects_hostname_with_mixed_public_and_private_results(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 0)),
            (2, 1, 6, "", ("10.0.0.1", 0)),
        ]
        with pytest.raises(ValueError, match="private IP"):
            validate_url_not_private("http://mixed.example.com/file")

    def test_rejects_missing_hostname(self):
        with pytest.raises(ValueError, match="Failed to parse hostname"):
            validate_url_not_private("not-a-url")

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    def test_rejects_unresolvable_hostname(self, mock_getaddrinfo):
        import socket

        mock_getaddrinfo.side_effect = socket.gaierror("Name resolution failed")
        with pytest.raises(ValueError, match="Failed to resolve hostname"):
            validate_url_not_private("http://nonexistent.invalid/file")
