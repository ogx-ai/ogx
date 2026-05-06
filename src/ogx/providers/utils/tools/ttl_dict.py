# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from threading import RLock
from typing import Any


class TTLDict(dict):
    """
    A dictionary with a ttl for each item
    """

    def __init__(self, ttl_seconds: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ttl_seconds = ttl_seconds
        self._expires: dict[Any, Any] = {}  # expires holds when an item will expire
        self._lock = RLock()

        if args or kwargs:
            for k, v in self.items():
                self.__setitem__(k, v)

    def __delitem__(self, key):
        with self._lock:
            del self._expires[key]
            super().__delitem__(key)

    def __setitem__(self, key, value):
        with self._lock:
            self._expires[key] = time.monotonic() + self.ttl_seconds
            super().__setitem__(key, value)

    def _is_expired(self, key):
        if key not in self._expires:
            return False
        return time.monotonic() > self._expires[key]

    def __getitem__(self, key):
        with self._lock:
            if self._is_expired(key):
                del self._expires[key]
                super().__delitem__(key)
                raise KeyError(f"{key} has expired and was removed")

            return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        try:
            _ = self[key]
            return True
        except KeyError:
            return False

    def __repr__(self):
        with self._lock:
            for key in self.keys():
                if self._is_expired(key):
                    del self._expires[key]
                    super().__delitem__(key)
            return f"TTLDict({self.ttl_seconds}, {super().__repr__()})"
