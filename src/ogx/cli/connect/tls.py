# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import ssl

import httpx2


def untrusted_certificate_client() -> httpx2.Client:
    """HTTP client that skips TLS certificate verification.

    Only ever constructed after the user approved it with --insecure for this
    invocation.
    """
    return httpx2.Client(verify=False)


def is_untrusted_certificate_error(error: BaseException) -> bool:
    """Whether a connection failure is a TLS certificate verification error.

    Follows both the explicit (__cause__) and implicit (__context__) exception
    chains, since httpx surfaces the ssl.SSLCertVerificationError via implicit
    chaining from its ConnectError.
    """
    cause: BaseException | None = error
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, ssl.SSLCertVerificationError):
            return True
        cause = cause.__cause__ or cause.__context__
    return False
