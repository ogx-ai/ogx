# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from . import NVIDIAConfig


def _is_nvidia_hosted(config: NVIDIAConfig) -> bool:
    return "integrate.api.nvidia.com" in str(config.base_url)
