# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
from typing import Any


def instantiate_class_type(fully_qualified_name: str) -> type[Any]:
    """Import and return a class from its fully qualified module path.

    Args:
        fully_qualified_name: Dotted path like 'package.module.ClassName'.

    Returns:
        The class object referenced by the fully qualified name.
    """
    module_name, class_name = fully_qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)  # type: ignore[no-any-return]
