# -*- coding: utf-8 -*-
"""
Weave: A modular PySide6 framework for the visual synthesis
and execution of high-concurrency simulation workflows.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: Apache-2.0 OR MIT

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
from weave.logger import get_logger

from .functions.version import __version__, commit, version_string

IMPORT_FAILURES = import_node_tree(__name__)

# Once, at import, so every log that carries a Silk problem also carries
# which Silk had it (G12). The submodule pin in the Weave checkout knows
# this too, and is no help at all to someone reading a log file.
get_logger("Silk").info(f"{version_string()} loaded")

__all__: list = ["__version__", "commit", "version_string"]
