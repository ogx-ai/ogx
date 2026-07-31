# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


def in_notebook():
    """Detect whether the current code is running inside a Jupyter notebook.

    Returns:
        True if running in a Jupyter/IPython notebook, False otherwise.
    """
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None or "IPKernelApp" not in ipython.config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
