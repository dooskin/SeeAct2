"""Compatibility shim for tests that import 'seeact_package.seeact.*'.

This exposes the real 'seeact' package under the 'seeact_package.seeact'
namespace when the sys.path points at the repository's 'seeact_package' root.
"""
import importlib
import sys

# Import the real 'seeact' package (located at this same filesystem level)
_real_seeact = importlib.import_module("seeact")

# Expose it under this package as a submodule so
# 'seeact_package.seeact.agent' resolves correctly.
sys.modules[__name__ + ".seeact"] = _real_seeact
