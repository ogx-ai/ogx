# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import ssl
from unittest.mock import MagicMock

from openai import APIConnectionError

from ogx.cli.connect.connect import ConnectParser


def parse_group_args(argv: list[str]) -> argparse.Namespace:
    """Parse argv (without the leading 'connect') through the real connect group.

    The group resolves args.connect_url exactly as the production CLI does, so
    per-agent command tests receive the same namespace the server dispatch gets.
    """
    top = argparse.ArgumentParser()
    ConnectParser(top.add_subparsers())
    return top.parse_args(["connect", *argv])


def make_untrusted_certificate_error() -> APIConnectionError:
    """An APIConnectionError whose implicit context chain holds an SSL cert error,

    mirroring how the OpenAI SDK wraps httpx.ConnectError wrapping
    ssl.SSLCertVerificationError for self-signed servers.
    """
    ssl_error = ssl.SSLCertVerificationError("[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate")
    connection_error = Exception("[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate")
    connection_error.__context__ = ssl_error
    api_error = APIConnectionError(request=MagicMock())
    api_error.__context__ = connection_error
    return api_error
