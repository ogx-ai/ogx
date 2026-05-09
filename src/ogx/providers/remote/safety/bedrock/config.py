# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.providers.utils.bedrock.config import BedrockBaseConfig
from ogx_api import json_schema_type


@json_schema_type
class BedrockSafetyConfig(BedrockBaseConfig):
    """Configuration for the AWS Bedrock safety provider."""

    pass
