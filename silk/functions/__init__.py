# -*- coding: utf-8 -*-
"""Silk agent runtime — Qt-free core (ported from the silk-playground project).

No module in this package imports PySide6 or touches Qt state: the ToolSet
hierarchy, the ToolBox registry with hard role enforcement, capabilities,
hooks, reflection, usage limits, typed stream events, the Role/RoleBinding
model, GraphEngine, and the AgentLoop autonomous run loop all run and
unit-test headless (no QApplication). Nodes under ``../nodes`` are thin
graph/Qt wiring over these functions (see WRITING_A_PLUGIN.md).

Note: importing this package *through the suite* (``weave.plugins.silk``)
runs node discovery in the parent ``__init__``, which imports Qt modules —
that is standard for every Weave suite and still needs no QApplication.
"""
