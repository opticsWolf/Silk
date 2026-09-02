# Reconstructing `ARCHITECTURE.md`

The files in this folder are the single-file document
`docs/ARCHITECTURE.md` split one top-level section per file. To get the
single file back — to diff it, release it, or keep editing it as one
document — merge the numbered files back together in manifest order.

`README.md` and this file are **not** part of the merge; only files matching
`NN-slug.md` are.

## Manifest (merge order)

| Order | File | Top-level section |
|---|---|---|
| 0 | `00-header.md` | title, intro, Contents |
| 1 | `01-layers.md` | Layers |
| 2 | `02-wiring.md` | Wiring at a glance |
| 3 | `03-protocols.md` | The two protocol contracts |
| 4 | `04-agent-loop.md` | The agent loop |
| 5 | `05-lifecycle-and-failure-semantics.md` | Lifecycle and failure semantics |
| 6 | `06-model-layer.md` | Model layer |
| 7 | `07-tool-transport.md` | Tool transport |
| 8 | `08-tool-system.md` | The tool system |
| 9 | `09-system-prompt-assembly.md` | System prompt assembly |
| 10 | `10-usage-reflection-validation.md` | Usage, reflection, and output validation |
| 11 | `11-task-system-signoff.md` | Task system and sign-off |
| 12 | `12-multi-agent.md` | Multi-agent |
| 13 | `13-tool-discovery.md` | Tool discovery and search |
| 14 | `14-presets.md` | Presets |
| 15 | `15-event-streams.md` | Event streams |
| 16 | `16-thread-model.md` | Thread model |
| 17 | `17-invariants.md` | Invariants |
| 18 | `18-design-rules.md` | Design rules |
| 19 | `19-where-new-behaviour-goes.md` | Where new behaviour goes |
| 20 | `20-graph-authoring.md` | Graph authoring |

The merge contract is the order above. Per-file line counts are
deliberately not tracked (they churn on every edit); the rebuild script
prints the resulting line count as a sanity check.

## Rebuild (canonical — Python)

Run from the repo root:

```python
import re
from pathlib import Path

d = Path("docs/architecture")
parts = sorted(p for p in d.glob("*.md") if re.match(r"\d{2}-", p.name))
text = "".join(p.read_text(encoding="utf-8") for p in parts)
text = re.sub(r"\]\((\d{2}-[a-z0-9-]+\.md)#([a-z0-9-]+)\)", r"](#\2)", text)
text = re.sub(r"\]\(\.\./([A-Za-z0-9_]+\.md)", r"](\1", text)
Path("docs/ARCHITECTURE.md").write_text(text, encoding="utf-8", newline="")
print("rebuilt docs/ARCHITECTURE.md:", len(text.splitlines()), "lines")
```

`newline=""` is load-bearing: `write_text` would otherwise translate `\n` to
the platform line ending (CRLF on Windows) and the file would no longer be
byte-identical to the LF canonical form. `read_text` already normalises
CRLF working-tree chunks back to `\n`, so the rebuild is LF on every
machine.

It does three things:

1. **Concatenates** `00-header.md` through `20-*.md` in order. The chunks
   were cut at section boundaries with their original spacing, so plain
   concatenation restores the document's line structure exactly.
2. **Reverts the split's link rewrites** — the only transformation the split
   applies:

   | In the split files | In the single file | Why the split rewrites it |
   |---|---|---|
   | `](NN-slug.md#anchor)` | `](#anchor)` | cross-section links must cross files here, but are same-file anchors in the single document |
   | `](../NAME.md)` (optionally with `#anchor`) | `](NAME.md)` (anchor preserved) | this folder sits one level deeper than `docs/` |
3. **Writes canonical LF** (`newline=""`), byte-identical to the pre-split
   `docs/ARCHITECTURE.md` as it lives in git.

With `core.autocrlf=true` the chunk files may show CRLF in the working tree
— that is normal and irrelevant: `read_text` normalises it away, and git
stores the LF form.

## Verify

```bash
python -c "import subprocess, pathlib; a = subprocess.run(['git','show','6f6bfeb:docs/ARCHITECTURE.md'], capture_output=True).stdout.decode('utf-8'); b = pathlib.Path('docs/ARCHITECTURE.md').read_text(encoding='utf-8'); print('IDENTICAL' if a.replace(chr(13)+chr(10), chr(10)) == b.replace(chr(13)+chr(10), chr(10)) else 'DIFFERS')"
```

(`6f6bfeb` is the last commit that contained the single file — the split
was committed as `e7d9669`. Note the single-file form post-dated that
commit by a set of accuracy fixes that went straight into the split files,
so a reconstruction of the *current* split files differs from `6f6bfeb`
by exactly those edits; for a byte-exact check compare against a
pre-split working-tree copy instead.)

## Quick rebuild (PowerShell, equivalent normalisation)

```powershell
$parts = Get-ChildItem docs\architecture\*.md | Where-Object Name -match '^\d{2}-' | Sort-Object Name
$text = -join ($parts | ForEach-Object { Get-Content $_.FullName -Raw })
$text = $text -replace '\]\((\d{2}-[a-z0-9-]+\.md)#([a-z0-9-]+)\)', '](#$2)'
$text = $text -replace '\]\(\.\./([A-Za-z0-9_]+\.md)', ']($1'
[IO.File]::WriteAllText((Join-Path (Get-Location) 'docs\ARCHITECTURE.md'), $text, (New-Object Text.UTF8Encoding($false)))
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
- **Line endings:** the canonical form in git is LF (with
  `core.autocrlf=true` the working tree shows CRLF — normal, and
  normalised away by both rebuild recipes). Always rebuild with the
  `newline=''` form above; never `Set-Content` (it forces CRLF).
