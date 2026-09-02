# Silk — Architecture

Silk is a Weave plugin that embeds a GGUF local-LLM agent into a visual
graph. This document describes how the pieces fit together, module by
module, based on the code in `functions/`, `nodes/`, and `widgets/`.

It is the companion to [NODES.md](../NODES.md) (what the graph nodes do) and
[TOOLS.md](../TOOLS.md) (the built-in tools and how to add your own).

This document is **split across the files in this folder** — one per
top-level section — so each can be read or edited in isolation. Cross-section
links use `](NN-slug.md#anchor)`; the single-file form of every link is
restored by the reconstruction below.

- [Reconstruct the single-file `ARCHITECTURE.md`](RECONSTRUCT.md) — merge all
  sections back into one file, byte-identical to the pre-split document
- `00-header.md` — the original title, intro, and Contents block; it is the
  first piece of the reconstruction (its Contents anchors are valid in the
  single file, not in this folder)

## Contents

- [Layers](01-layers.md)
- [Wiring at a glance](02-wiring.md)
- [The two protocol contracts](03-protocols.md)
- [The agent loop](04-agent-loop.md)
- [Lifecycle and failure semantics](05-lifecycle-and-failure-semantics.md)
- [Model layer](06-model-layer.md)
- [Tool transport](07-tool-transport.md)
- [The tool system](08-tool-system.md)
  - [ToolBox](08-tool-system.md#toolbox)
  - [Capabilities](08-tool-system.md#capabilities)
  - [ToolSet layer](08-tool-system.md#toolset-layer)
  - [Roles](08-tool-system.md#roles)
  - [Hooks and middleware](08-tool-system.md#hooks-and-middleware)
  - [Hook catalog](08-tool-system.md#hook-catalog)
- [System prompt assembly](09-system-prompt-assembly.md)
- [Usage, reflection, and output validation](10-usage-reflection-validation.md)
- [Task system and sign-off](11-task-system-signoff.md)
  - [The approval gate](11-task-system-signoff.md#functionsapprovalpy--the-approval-gate)
  - [The decision seam](11-task-system-signoff.md#functionsdecision_seampy--asking-from-a-worker-thread)
  - [Grants](11-task-system-signoff.md#functionsgrantspy--dont-ask-again)
- [Multi-agent](12-multi-agent.md)
- [Tool discovery and search](13-tool-discovery.md)
- [Presets](14-presets.md)
- [Event streams](15-event-streams.md)
- [Thread model](16-thread-model.md)
- [Invariants](17-invariants.md)
- [Design rules](18-design-rules.md)
- [Where new behaviour goes](19-where-new-behaviour-goes.md)
- [Graph authoring](20-graph-authoring.md)

## Files

| File | Contents |
|---|---|
| `00-header.md` | title, intro, and the single-file Contents block |
| `01-layers.md` | Layers |
| `02-wiring.md` | Wiring at a glance |
| `03-protocols.md` | The two protocol contracts |
| `04-agent-loop.md` | The agent loop |
| `05-lifecycle-and-failure-semantics.md` | Lifecycle and failure semantics |
| `06-model-layer.md` | Model layer |
| `07-tool-transport.md` | Tool transport |
| `08-tool-system.md` | The tool system |
| `09-system-prompt-assembly.md` | System prompt assembly |
| `10-usage-reflection-validation.md` | Usage, reflection, and output validation |
| `11-task-system-signoff.md` | Task system and sign-off |
| `12-multi-agent.md` | Multi-agent |
| `13-tool-discovery.md` | Tool discovery and search |
| `14-presets.md` | Presets |
| `15-event-streams.md` | Event streams |
| `16-thread-model.md` | Thread model |
| `17-invariants.md` | Invariants |
| `18-design-rules.md` | Design rules |
| `19-where-new-behaviour-goes.md` | Where new behaviour goes |
| `20-graph-authoring.md` | Graph authoring |
| `README.md` | this index (not part of the reconstruction) |
| `RECONSTRUCT.md` | how to remerge everything into one file |
