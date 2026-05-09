# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re

CONVERSATION_ID_PATTERN = re.compile(r"^conv_[0-9a-f]{48}$")
