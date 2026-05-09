# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any


class Subcommand:
    """All llama cli subcommands must inherit from this class"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> "Subcommand":
        return cls(*args, **kwargs)

    def _add_arguments(self) -> None:
        pass
