# -*- coding: utf-8 -*-
"""The public web as agent tools: gossamer search, fetch and extraction.

Distilled from the bundled skill ``gossamer/skills/gossamer/SKILL.md``
(served upstream as ``skills://gossamer``) and ``docs/QUICKREF.md``. The
descriptions below are the skill's routing table, budgets and gotchas
compressed to what the model needs at call time; re-sync them when the
skill changes (upstream guards its drift with ``test_skill_drift.py``).

What was taken from the skill, per tool:

``web_search``
    Cheap provider search first (``search_only=true``), fetch later --
    skill "Budgets". Points at ``research_by_category`` for domain
    questions -- skill "Routing". Keyed-provider failures name the exact
    variable: surface, don't retry -- skill "Auth".
``research_by_category``
    The routing table itself (scholarly -> OpenAlex, legal ->
    courtlistener/oldp/hudoc, patent -> EPO/KIPRIS/patentsview/lens
    key-gated, financial -> yahoo/frankfurter/eurostat/..., geo ->
    open-meteo/overpass, else DuckDuckGo). ``filter``/``select``/
    ``title``/``author`` are OpenAlex-only and rejected otherwise.
``inspect_html_page``
    ``check_sources`` first for shaky URLs; ``query`` slices long pages;
    paging re-reads the cached full text -- skill "Budgets" + "Cache
    behavior". No ``use_smart`` knob: the browser extra is not installed
    here, so only static fetch runs (skill "Budgets" caveat).
``check_sources``
    Never cached, liveness only -- skill "Cache behavior". ``status``
    HEAD probe, ``content`` GET for HEAD-rejecting hosts.
``extract_document``
    ``pages`` ranges for large PDFs (1-based inclusive, ``10-20`` style);
    JATS-first scholarly route (``pages=`` errors on JATS-resolvable
    sources, math renders correctly); tables as markdown -- skill
    "Documents". No ``store`` knob: text only, nothing written outside
    the sandbox. gossamer opens any local path it is given, so the pack
    passes URLs and bare DOIs through and jails everything else to the
    file sandbox (read grant, root, denied paths) before the call.
``web_download``
    gossamer has no configured download directory -- ``output_path`` is
    per-call by design -- so the pack supplies the ceiling instead: a
    bare filename is joined onto ``<sandbox>/downloads/`` and anything
    else is refused. Atomic stream, byte cap, PDF signature check,
    resume, robots/SSRF enforced -- skill "Documents" + QUICKREF.

Egress: every tool sends the query or URL off-machine. The group stays
out of ``DEFAULT_GROUPS``, so a fresh box leaves these unticked -- the
tick is the grant, the same rule as the classifier's air-gap switch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError

from weave.logger import get_logger

log = get_logger("SilkWebTools")

#: The tools this pack registers, in the order an agent reads them:
#: find, then verify, then read -- download last, it writes.
WEB_TOOLS = (
    "web_search",
    "research_by_category",
    "check_sources",
    "inspect_html_page",
    "extract_document",
    "web_download",
)

#: Characters of a result handed back to the model.
OUTPUT_LIMIT = 20_000

#: Where the toolbox keeps its HTTP + parse cache when the environment
#: does not say otherwise (same precedence the skill documents:
#: explicit env first, then a Silk-owned default -- never CWD-relative).
CACHE_DIR = os.path.expanduser(
    os.environ.get("GOSSAMER_CACHE_DIR", "~/.weave/silk/gossamer_cache")
)

try:
    from gossamer import (  # noqa: F401
        ToolboxConfig as _ToolboxConfig,
        WebResearcherToolbox as _WebResearcherToolbox,
    )
    _GOSSAMER_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 -- Rust-linkage fails like this too
    _ToolboxConfig = None  # type: ignore[assignment]
    _WebResearcherToolbox = None  # type: ignore[assignment]
    _GOSSAMER_ERROR = str(exc)

_TOOLBOX: Any = None
_TOOLBOX_LOCK = threading.Lock()
_CALL_LOCK = threading.Lock()


def _toolbox() -> Any:
    """The process-wide toolbox, built on first use (never at attach).

    The group catalog is probed by running every attacher against a
    scratch box where nothing may execute -- construction happens here,
    on the first real call, inside the lock.
    """
    global _TOOLBOX
    if _TOOLBOX is not None:
        return _TOOLBOX
    with _TOOLBOX_LOCK:
        if _TOOLBOX is not None:
            return _TOOLBOX
        _TOOLBOX = _WebResearcherToolbox(
            _ToolboxConfig(cache_dir=CACHE_DIR, max_tokens=4000)
        )
        return _TOOLBOX


def _safe_filename(name: str) -> str:
    """A bare filename or a refusal -- the sandbox ceiling in one place.

    The model supplies a name, never a path: separators, parent
    references and absolute paths are rejected here, and the sandbox's
    own ``resolve_path`` re-checks the joined target as a backstop.
    """
    text = (name or "").strip()
    if not text or text in (".", "..") or Path(text).name != text:
        raise ValueError(
            f"filename must be a bare file name, got {name!r}: no "
            "directories, no '..', no absolute paths"
        )
    return text


#: A whole-string bare DOI (``10.1234/x`` or ``doi:10.1234/x``). Stricter
#: than gossamer's own probe, which finds a DOI *inside* any string -- a
#: local path like ``C:/x/10.1234/a.pdf`` included.
_BARE_DOI = re.compile(r"^(?:doi:\s*)?10\.\d{4,9}/\S+$", re.IGNORECASE)


def _document_source(source: str, sandbox: Any) -> str:
    """The source gossamer may open: remote as given, local only if jailed.

    gossamer reads any local path it is handed, so everything that is not
    an http(s) URL or a bare DOI goes through the file sandbox first --
    the same read grant, root and denied paths the file tools obey -- and
    reaches gossamer as the resolved absolute path.
    """
    text = source.strip()
    if urlparse(text).scheme.lower() in ("http", "https"):
        return text
    if _BARE_DOI.match(text) and not Path(text).exists():
        return text
    sandbox.check_read()
    path = sandbox.resolve_path(text)
    if not path.is_file():
        raise ValueError(f"no file at {text!r} in the sandbox")
    sandbox.check_extension(path)
    return str(path)


def _format(raw: str) -> str:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = raw
    text = (
        json.dumps(data, ensure_ascii=False, default=str, indent=1)
        if not isinstance(data, str) else data
    )
    if len(text) > OUTPUT_LIMIT:
        text = text[:OUTPUT_LIMIT] + f"\n… (truncated at {OUTPUT_LIMIT} chars)"
    return text


def _run(name: str, args: Dict[str, Any]) -> str:
    try:
        box = _toolbox()
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        return (
            f"error: web backend unavailable: {exc}\n"
            "remedy: install gossamer-web for the Weave environment "
            "(uv pip install gossamer-web, or maturin develop from a "
            "checkout) and retry"
        )
    try:
        with _CALL_LOCK:
            raw = box.execute_tool(name, args)
    except Exception as exc:  # noqa: BLE001 -- tools return errors
        return f"error: {name} failed: {exc}"
    return _format(raw)


class WebSearchArgs(BaseModel):
    query: str = Field(..., min_length=1, description="The search topic.")
    search_only: bool = Field(
        True,
        description=(
            "true: provider hits only -- cheap, with citeable links. "
            "false: also fetch up to `depth` pages as markdown -- slower "
            "and token-heavy; prefer true, then inspect the winners."
        ),
    )
    max_results: int = Field(
        5, ge=1, le=10, description="Provider hits to consider."
    )
    depth: int = Field(
        3, ge=1, le=5,
        description="Pages fetched when search_only=false; keep small.",
    )
    max_tokens: int = Field(
        4000, ge=0, le=8000, description="Token budget; 0 = toolbox default."
    )
    provider: Optional[str] = Field(
        None,
        description=(
            "Force a provider (default duckduckgo). google/bing/exa need "
            "keys -- a keyed failure names the exact variable: surface it "
            "to the user instead of retrying."
        ),
    )


class ResearchArgs(BaseModel):
    query: str = Field(
        ..., min_length=1,
        description="Short keywords for Hub-backed providers, not sentences.",
    )
    category: Optional[Literal[
        "scholarly", "legal", "patent", "financial", "geo", "general",
    ]] = Field(None, description="Omit to auto-route the query.")
    provider: Optional[str] = Field(
        None, description="Force a provider inside the category."
    )
    max_results: int = Field(5, ge=1, le=10)
    filter: Optional[str] = Field(
        None, description="OpenAlex-native filter; rejected otherwise."
    )
    select: Optional[str] = Field(
        None, description="OpenAlex-native field projection; rejected otherwise."
    )
    title: Optional[str] = Field(
        None, description="OpenAlex title.search; rejected otherwise."
    )
    author: Optional[str] = Field(
        None, description="OpenAlex author search; rejected otherwise."
    )


class CheckArgs(BaseModel):
    urls: List[str] = Field(
        ..., min_length=1, max_length=20,
        description="Up to 20 URLs to probe.",
    )
    mode: Literal["status", "content"] = Field(
        "status",
        description="'status': HEAD probe. 'content': GET for hosts that reject HEAD.",
    )


class InspectArgs(BaseModel):
    url: str = Field(..., min_length=1, description="The page to read.")
    query: Optional[str] = Field(
        None,
        description="Focus slice: only the most relevant sections return.",
    )
    offset: int = Field(
        0, ge=0,
        description="Char offset into the cached markdown; follow next_offset to continue.",
    )
    max_chunks: int = Field(1, ge=1, le=5)
    max_tokens: int = Field(
        4000, ge=0, le=8000, description="Token budget; 0 = toolbox default."
    )


class DownloadArgs(BaseModel):
    source: str = Field(..., min_length=1, description="The file URL.")
    filename: str = Field(
        ..., min_length=1,
        description=(
            "Bare file name only -- saved as <sandbox>/downloads/<name> "
            "for the file tools to read. No directories, no '..'."
        ),
    )
    expected_format: Literal["auto", "pdf"] = Field(
        "auto", description="'pdf' enforces the PDF signature."
    )
    overwrite: bool = Field(False)
    resume: bool = Field(
        False, description="Continue an existing partial file."
    )
    max_bytes: int = Field(
        0, ge=0, description="Byte cap; 0 = the configured response cap."
    )


class ExtractArgs(BaseModel):
    source: str = Field(
        ..., min_length=1,
        description=(
            "An http(s) URL, a bare DOI, or a file path inside the "
            "sandbox (relative to its root; needs read access)."
        ),
    )
    pages: Optional[str] = Field(
        None,
        description=(
            "PDF '10-20', 1-based inclusive (sheets for XLSX). Large PDFs: "
            "page here instead of the whole file. Errors on JATS-routed "
            "scholarly sources -- those have sections, not pages."
        ),
    )
    tables_as: Literal["json", "markdown", "csv"] = Field(
        "json", description="Spreadsheet rendering; body tables are markdown regardless."
    )


def attach_web_tools(toolbox: Any, sandbox: Any) -> None:
    """Register the public-web tools. Recipe-shaped like every group.

    *sandbox* is the ceiling downloads are jailed to -- the one tool
    that takes it, rather than ignores it.
    """
    if _WebResearcherToolbox is None:
        log.info(
            "web tools unavailable (%s); install gossamer-web for the "
            "Weave environment to offer them", _GOSSAMER_ERROR,
        )
        return

    def run(name: str, args: BaseModel) -> str:
        return _run(name, args.model_dump(exclude_none=True))

    @toolbox.register(
        name="web_search",
        description=(
            "Search the public web (sends the query off-machine) and "
            "optionally fetch the top hits as markdown. Prefer "
            "research_by_category when the question is papers, case law, "
            "patents, market data or places -- it routes to the right "
            "provider. Results are cached per query for the TTL."
        ),
        args_model=WebSearchArgs,
        tags=("web", "search", "gossamer"), category="web", risk="low",
    )
    def web_search(_db: Any, _session: Any, **args: Any) -> str:
        return run("web_search", WebSearchArgs(**args))

    @toolbox.register(
        name="research_by_category",
        description=(
            "Classify-first research (sends the query off-machine): "
            "scholarly->OpenAlex, legal->courtlistener/oldp/hudoc, "
            "patent->EPO/KIPRIS/patentsview/lens, "
            "financial->yahoo/frankfurter/eurostat, geo->open-meteo/overpass, "
            "else general web search. Patent and some providers are "
            "key-gated and fail fast naming the exact variable -- surface "
            "it, don't retry. Papers: research, then check_sources the "
            "candidates, then inspect or extract the winners."
        ),
        args_model=ResearchArgs,
        tags=("web", "research", "gossamer"), category="web", risk="low",
    )
    def research_by_category(_db: Any, _session: Any, **args: Any) -> str:
        return run("research_by_category", ResearchArgs(**args))

    @toolbox.register(
        name="check_sources",
        description=(
            "Probe whether URLs are reachable before spending a fetch on "
            "them. Never cached -- it probes liveness, so a repeat may "
            "differ."
        ),
        args_model=CheckArgs,
        tags=("web", "liveness", "gossamer"), category="web", risk="low",
    )
    def check_sources(_db: Any, _session: Any, **args: Any) -> str:
        return run("check_sources", CheckArgs(**args))

    @toolbox.register(
        name="inspect_html_page",
        description=(
            "Fetch one page as markdown with follow-up links and "
            "provenance (static fetch; JS-walled pages may fail). Reads "
            "are cached and carry cache_hit; paging re-reads the cached "
            "full text, so re-reads stay cheap."
        ),
        args_model=InspectArgs,
        tags=("web", "fetch", "gossamer"), category="web", risk="low",
    )
    def inspect_html_page(_db: Any, _session: Any, **args: Any) -> str:
        return run("inspect_html_page", InspectArgs(**args))

    @toolbox.register(
        name="extract_document",
        description=(
            "Extract text and tables from a document URL, DOI, or a file "
            "inside the sandbox (PDF/DOCX/XLSX/PPTX/feeds). Scholarly sources prefer "
            "publisher full text when DOI-resolvable -- math then renders "
            "correctly. Large PDFs: page with `pages`, and overlong output "
            "is truncated."
        ),
        args_model=ExtractArgs,
        tags=("web", "extract", "gossamer"), category="web", risk="low",
    )
    def extract_document(_db: Any, _session: Any, **args: Any) -> str:
        parsed = ExtractArgs(**args)
        try:
            parsed.source = _document_source(parsed.source, sandbox)
        except (PermissionError, ValueError, OSError) as exc:
            return (
                f"error: extract refused: {exc}\nremedy: pass an http(s) "
                "URL or a DOI, or a file inside the sandbox (web_download "
                "puts remote files in downloads/); local files need a "
                "read tool ticked (e.g. read_file)"
            )
        return run("extract_document", parsed)

    @toolbox.register(
        name="web_download",
        description=(
            "Save a remote file as <sandbox>/downloads/<filename> "
            "(filename only -- the sandbox root is the ceiling) for the "
            "file tools to read. Atomic stream, byte-capped, PDF "
            "signature-checked; resume continues a partial; robots/SSRF "
            "enforced, bot walls never bypassed."
        ),
        args_model=DownloadArgs,
        tags=("web", "download", "gossamer"), category="web", risk="medium",
    )
    def web_download(_db: Any, _session: Any, **args: Any) -> str:
        try:
            parsed = DownloadArgs(**args)
            name = _safe_filename(parsed.filename)
        except (ValidationError, ValueError) as exc:
            return f"error: bad download request: {exc}"
        try:
            sandbox.check_write()
            target = sandbox.resolve_path(str(Path("downloads") / name))
            sandbox.assert_writable(target)
        except (PermissionError, ValueError) as exc:
            return (
                f"error: download refused: {exc}\nremedy: downloads "
                "need write access -- tick a write tool (e.g. write_file) "
                "to grant it, then retry"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        out = _run("download_file", {
            "source": parsed.source,
            "output_path": str(target),
            "expected_format": parsed.expected_format,
            "overwrite": parsed.overwrite,
            "resume": parsed.resume,
            "max_bytes": parsed.max_bytes,
        })
        if out.startswith("error:"):
            return out
        rel = target.relative_to(sandbox.root_dir).as_posix()
        return f"saved: {rel} (sandbox-relative)\n" + out
