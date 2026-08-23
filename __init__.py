# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0

Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.

Nodes (under ``nodes/``): GGUF model loader (llama.cpp pool), Silk ToolBox
(sandboxed file/search tools with hard role enforcement), Silk Role
(declarative agent configuration), Silk Agent (the autonomous tool-calling
loop with Exec trigger/done ports for chaining agent networks), and a chat
log display.

The Qt-free runtime — ToolSet/ToolBox, capabilities, hooks, Role/RoleBinding,
AgentLoop, GraphEngine — lives under ``functions/`` and is importable without
PySide6. Requires the optional ``llama-cpp-python`` dependency for inference;
the nodes import without it but the loader logs an error. This suite
supersedes the former ``weave.plugins.llm`` draft.

Importing this package registers the nodes with ``NODE_REGISTRY``.
"""

from weave._discovery import import_node_tree

IMPORT_FAILURES = import_node_tree(__name__)

__all__: list = []
