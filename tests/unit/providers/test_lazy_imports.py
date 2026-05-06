# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tests to verify that provider modules do not eagerly load heavy dependencies.

Each test uses subprocess isolation to check that importing a module
does not pull in heavy libraries (torch, numpy, etc.) until they are
explicitly needed.
"""

import subprocess
import sys


def _check_module_import_isolation(module_path: str, forbidden_modules: list[str]) -> dict:
    """
    Run a subprocess to import a module and check which forbidden modules are loaded.

    Returns a dict with 'loaded' (list of unexpectedly loaded modules) and 'success' (bool).
    """
    check_script = f"""
import sys

# Record modules before import
before = set(sys.modules.keys())

# Import the target module
{module_path}

# Check which forbidden modules were loaded
after = set(sys.modules.keys())
new_modules = after - before

forbidden = {forbidden_modules!r}
loaded = [m for m in forbidden if any(m == mod or mod.startswith(m + '.') for mod in new_modules)]

# Output result
import json
print(json.dumps({{"loaded": loaded, "new_count": len(new_modules)}}))
"""

    result = subprocess.run(
        [sys.executable, "-c", check_script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        return {"loaded": [], "error": result.stderr, "success": False}

    import json

    output = json.loads(result.stdout.strip())
    output["success"] = True
    return output


class TestPromptGuardLazyImports:
    """Test that prompt_guard safety provider doesn't load torch or transformers at import time."""

    def test_no_torch_transformers_on_import(self):
        """Verify prompt_guard module import doesn't load torch or transformers."""
        result = _check_module_import_isolation(
            "from ogx.providers.inline.safety.prompt_guard import prompt_guard",
            ["torch", "transformers"],
        )

        assert result.get("success"), f"Import failed: {result.get('error', 'unknown error')}"
        assert not result["loaded"], (
            f"Heavy modules loaded unexpectedly during prompt_guard import: {result['loaded']}. "
            "These should be lazily loaded only when initialize() is called."
        )


def _check_no_forbidden_imports(module_path: str, forbidden: list[str]) -> tuple[bool, str]:
    """Import a module in a subprocess and check that forbidden modules are not loaded."""
    code = f"""
import sys
import importlib
importlib.import_module("{module_path}")
loaded = [m for m in {forbidden!r} if m in sys.modules]
if loaded:
    print("FORBIDDEN:" + ",".join(loaded))
else:
    print("OK")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    output = result.stdout.strip()
    if output.startswith("FORBIDDEN:"):
        return False, output.split(":", 1)[1]
    return True, ""


class TestEmbeddingMixinLazyImports:
    """Verify embedding_mixin.py does not eagerly import torch."""

    def test_no_torch_on_import(self):
        ok, loaded = _check_no_forbidden_imports(
            "ogx.providers.utils.inference.embedding_mixin",
            ["torch"],
        )
        assert ok, f"embedding_mixin.py eagerly loaded: {loaded}"


class TestFaissLazyImports:
    """Verify faiss.py does not eagerly import faiss or numpy."""

    def test_no_faiss_numpy_on_import(self):
        ok, loaded = _check_no_forbidden_imports(
            "ogx.providers.inline.vector_io.faiss.faiss",
            ["faiss", "numpy"],
        )
        assert ok, f"faiss.py eagerly loaded: {loaded}"


class TestSqliteVecLazyImports:
    """Verify sqlite_vec.py does not eagerly import numpy or sqlite_vec."""

    def test_no_numpy_sqlite_vec_on_import(self):
        ok, loaded = _check_no_forbidden_imports(
            "ogx.providers.inline.vector_io.sqlite_vec.sqlite_vec",
            ["numpy", "sqlite_vec"],
        )
        assert ok, f"sqlite_vec.py eagerly loaded: {loaded}"


class TestVectorStoreLazyImports:
    """Verify vector_store.py does not eagerly import numpy."""

    def test_no_numpy_on_import(self):
        ok, loaded = _check_no_forbidden_imports(
            "ogx.providers.utils.memory.vector_store",
            ["numpy"],
        )
        assert ok, f"vector_store.py eagerly loaded: {loaded}"
