# Reconstructing `ARCHITECTURE.md`

The files in this folder are the single-file document
`docs/ARCHITECTURE.md` split one top-level section per file. To get the
single file back — to diff it, release it, or keep editing it as one
document — merge the numbered files back together in manifest order.

`README.md` and this file are **not** part of the merge; only files matching
`NN-slug.md` are.

## Manifest (merge order)

| Order | File | Top-level section | Lines |
|---|---|---|---|
| 0 | `00-header.md` | title, intro, Contents | 37 |
| 1 | `01-layers.md` | Layers | 19 |
| 2 | `02-wiring.md` | Wiring at a glance | 71 |
| 3 | `03-protocols.md` | The two protocol contracts | 32 |
| 4 | `04-agent-loop.md` | The agent loop | 102 |
| 5 | `05-lifecycle-and-failure-semantics.md` | Lifecycle and failure semantics | 54 |
| 6 | `06-model-layer.md` | Model layer | 42 |
| 7 | `07-tool-transport.md` | Tool transport | 32 |
| 8 | `08-tool-system.md` | The tool system | 276 |
| 9 | `09-system-prompt-assembly.md` | System prompt assembly | 28 |
| 10 | `10-usage-reflection-validation.md` | Usage, reflection, and output validation | 26 |
| 11 | `11-task-system-signoff.md` | Task system and sign-off | 69 |
| 12 | `12-multi-agent.md` | Multi-agent | 67 |
| 13 | `13-tool-discovery.md` | Tool discovery and search | 15 |
| 14 | `14-presets.md` | Presets | 12 |
| 15 | `15-event-streams.md` | Event streams | 32 |
| 16 | `16-thread-model.md` | Thread model | 20 |
| 17 | `17-invariants.md` | Invariants | 28 |
| 18 | `18-design-rules.md` | Design rules | 21 |
| 19 | `19-where-new-behaviour-goes.md` | Where new behaviour goes | 16 |

The pre-split document is **999 lines**; a correct reconstruction
reproduces it exactly (the per-file line counts above sum to it).

## Rebuild (canonical — Python)

Run from the repo root:

```python
import re
from pathlib import Path

d = Path("docs/architecture")
parts = sorted(p for p in d.glob("*.md") if re.match(r"\d{2}-", p.name))
text = "".join(p.read_text(encoding="utf-8") for p in parts)
text = re.sub(r"\]\((\d{2}-[a-z0-9-]+\.md)#([a-z0-9-]+)\)", r"](#\2)", text)
text = (text.replace("](../NODES.md)", "](NODES.md)")
          .replace("](../TOOLS.md)", "](TOOLS.md)")
          .replace("](../OPEN_TOPICS.md)", "](OPEN_TOPICS.md)"))
Path("docs/ARCHITECTURE.md").write_text(text, encoding="utf-8")
print("rebuilt docs/ARCHITECTURE.md:", len(text.splitlines()), "lines")
```

It does two things:

1. **Concatenates** `00-header.md` through `19-*.md` in order. The chunks
   were cut at section boundaries with their original spacing, so plain
   concatenation restores the document's line structure exactly.
2. **Reverts the split's link rewrites** — the only transformation the split
   applies:

   | In the split files | In the single file | Why the split rewrites it |
   |---|---|---|
   | `](NN-slug.md#anchor)` | `](#anchor)` | cross-section links must cross files here, but are same-file anchors in the single document |
   | `](../NODES.md)`, `](../TOOLS.md)`, `](../OPEN_TOPICS.md)` | `](NODES.md)`, `](TOOLS.md)`, `](OPEN_TOPICS.md)` | this folder sits one level deeper than `docs/` |

The result is byte-identical to the pre-split `docs/ARCHITECTURE.md`.

## Verify

```bash
python -c "import subprocess, pathlib; a = subprocess.run(['git','show','HEAD:docs/ARCHITECTURE.md'], capture_output=True).stdout.decode('utf-8'); b = pathlib.Path('docs/ARCHITECTURE.md').read_text(encoding='utf-8'); print('IDENTICAL' if a.replace(chr(13)+chr(10), chr(10)) == b.replace(chr(13)+chr(10), chr(10)) else 'DIFFERS')"
```

(If the split was already committed, compare against the tagged/known-good
commit instead of `HEAD`.)

## Quick rebuild (PowerShell, equivalent normalisation)

```powershell
$parts = Get-ChildItem docs\architecture\*.md | Where-Object Name -match '^\d{2}-' | Sort-Object Name
$text = -join ($parts | ForEach-Object { Get-Content $_.FullName -Raw })
$text = $text -replace '\]\((\d{2}-[a-z0-9-]+\.md)#([a-z0-9-]+)\)', '](#$2)'
$text = $text -replace '\]\(\./NODES\.md\)', '](NODES.md)'
$text = $text -replace '\]\(\./TOOLS\.md\)', '](TOOLS.md)'
$text = $text -replace '\]\(\./OPEN_TOPICS\.md\)', '](OPEN_TOPICS.md)'
Set-Content -Encoding utf8 docs\ARCHITECTURE.md $text
```

## Maintenance rules

- **New top-level section:** take the next free number (`NN-slug.md`), add a
  row to the manifest above, and add the bullet to **both** Contents blocks
  (`00-header.md` — single-file form with `#anchor` — and `README.md` —
  folder form with `NN-slug.md` links).
- **Never renumber** existing files. The `NN-` prefix plus this manifest is
  the merge contract; renumbering silently breaks the reconstruction.
- **Follow the split's link conventions** in any new section so the
  normalisation above keeps working: cross-section links as
  `](NN-slug.md#anchor)`, sibling docs as `](../NAME.md)`.
