# -*- coding: utf-8 -*-
"""Bootstrap for the tests that re-run themselves as a plain script.

The "real Qt" checks (grant manager, canvas) spawn
``python <this test file>`` so a Qt object cycle can never take the
pytest process down with it. The subprocess has no pytest and no
conftest, so it must recreate what ``tests/conftest.py`` does: find the
Weave checkout for ``import weave``, install *this* checkout as the
package ``silk`` whatever its directory is named (the tests import
``silk.*``, never ``weave.plugins.silk`` — see conftest.py), and render
offscreen.

Import this module before the first ``silk.`` import. Under pytest it is
a no-op (conftest has already done the work); in the subprocess it does
all of it.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def bootstrap() -> None:
    # Find the Weave checkout: side-by-side clones first (dev layout),
    # then the submodule ancestor (as shipped).
    sibling = _ROOT.parent / "Weave"
    if (sibling / "weave" / "__init__.py").is_file():
        root = str(sibling)
    else:
        root = None
        for parent in _ROOT.parents:
            if (parent / "weave" / "__init__.py").is_file():
                root = str(parent)
                break
    if root and root not in sys.path:
        sys.path.insert(0, root)

    # Install this checkout as package `silk`, whatever the directory is
    # named ("Silk" standalone, "silk" as the submodule).
    if "silk" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "silk", _ROOT / "__init__.py", submodule_search_locations=[str(_ROOT)]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["silk"] = mod
        spec.loader.exec_module(mod)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


bootstrap()
