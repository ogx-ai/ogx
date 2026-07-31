# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from urllib.parse import urlparse

from . import NVIDIAConfig

_NVIDIA_HOSTED_HOSTNAME = "integrate.api.nvidia.com"


def _is_nvidia_hosted(config: NVIDIAConfig) -> bool:
    hostname = urlparse(str(config.base_url)).hostname
    return hostname == _NVIDIA_HOSTED_HOSTNAME
