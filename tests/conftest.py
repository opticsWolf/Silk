# -*- coding: utf-8 -*-
"""Silk's own test suite.

Layout note: this is the runtime half of the suite that used to live in
the Weave tree — it moved here 2026-09-09 (OPEN_TOPICS G4). The 13 seam
tests that drive a real ``Canvas``, ``weave.registry`` or the Weave
shutdown registry stayed in Weave's ``tests/``: they pin the Weave↔Silk
contract and rely on Weave's conftest Qt teardown barrier.

Bootstrap, in order:

1. **Find a Weave checkout** and put its root on ``sys.path`` so
   ``import weave`` resolves. ``functions/`` deliberately depends on
   ``weave.logger`` and the internals contract (D83, ``weave_contract``),
   and ``silk/__init__`` imports ``weave._discovery``. Two supported
   layouts: side-by-side clones (``Python/Silk`` + ``Python/Weave``, the
   dev checkout) and the submodule (``Weave/weave/plugins/silk``, as
   shipped). A standalone clone with neither needs Weave installed or on
   ``PYTHONPATH``.
2. **Import the repo root as the package ``silk``,** whatever the
   checkout's directory is named ("Silk" standalone, "silk" as the
   submodule). The subpackages cross-import within the plugin package
   (``from ..functions import ...`` in ``nodes/``), so the tests must
   import through a package identity whose ``__path__`` is the repo
   root — never through ``weave.plugins.silk``, which would silently
   test the host's submodule pin instead of this checkout.
3. **Qt renders offscreen** by default (the widget tests build real
   docks).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _weave_root() -> str | None:
    sibling = _ROOT.parent / "Weave"
    if (sibling / "weave" / "__init__.py").is_file():
        return str(sibling)
    for parent in _ROOT.parents:
        if (parent / "weave" / "__init__.py").is_file():
            return str(parent)
    return None


_w = _weave_root()
if _w and _w not in sys.path:
    sys.path.insert(0, _w)

# Import the repo root as the package `silk` (see bootstrap note 2).
_spec = importlib.util.spec_from_file_location(
    "silk", _ROOT / "__init__.py", submodule_search_locations=[str(_ROOT)]
)
_silk = importlib.util.module_from_spec(_spec)
sys.modules["silk"] = _silk
_spec.loader.exec_module(_silk)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
