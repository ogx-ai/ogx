# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import pathlib


class MessageAttachment:
    # https://developer.mozilla.org/en-US/docs/Glossary/Base64
    @classmethod
    def base64(cls, file_path: str) -> str:
        path = pathlib.Path(file_path)
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    # https://developer.mozilla.org/en-US/docs/Web/URI/Schemes/data
    @classmethod
    def data_url(cls, media_type: str, file_path: str) -> str:
        return f"data:{media_type};base64,{cls.base64(file_path)}"
