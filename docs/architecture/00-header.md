# Silk — Architecture

Silk is a Weave plugin that embeds a GGUF local-LLM agent into a visual
graph. This document describes how the pieces fit together, module by
module, based on the code in `functions/`, `nodes/`, and `widgets/`.

It is the companion to [NODES.md](../NODES.md) (what the graph nodes do) and
[TOOLS.md](../TOOLS.md) (the built-in tools and how to add your own).

## Contents

- [Layers](#layers)
- [Wiring at a glance](#wiring-at-a-glance)
- [The two protocol contracts](#the-two-protocol-contracts)
- [The agent loop](#the-agent-loop)
- [Lifecycle and failure semantics](#lifecycle-and-failure-semantics)
- [Model layer](#model-layer)
- [Tool transport](#tool-transport)
- [The tool system](#the-tool-system)
  - [ToolBox](#toolbox)
  - [Capabilities](#capabilities)
  - [ToolSet layer](#toolset-layer)
  - [Roles](#roles)
  - [Hooks and middleware](#hooks-and-middleware)
  - [Hook catalog](#hook-catalog)
- [System prompt assembly](#system-prompt-assembly)
- [Usage, reflection, and output validation](#usage-reflection-and-output-validation)
- [Task system and sign-off](#task-system-and-sign-off)
- [Multi-agent](#multi-agent)
- [Tool discovery and search](#tool-discovery-and-search)
- [Event streams](#event-streams)
- [Presets](#presets)
- [Thread model](#thread-model)
- [Invariants](#invariants)
- [Design rules](#design-rules)
- [Where new behaviour goes](#where-new-behaviour-goes)
- [Graph authoring](#graph-authoring--the-agent-places-nodes)
- [Self-modification](#self-modification--the-agent-extends-weave)

