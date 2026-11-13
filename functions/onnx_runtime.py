# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

What ONNX Runtime the ONNX nodes will get -- asked in a child process.

Weave's process must never import ``onnxruntime`` (and normally does not
have ``embroider`` installed): a second ONNX Runtime beside the one a
plugin already loaded can take the interpreter down, and the import
alone costs a session's startup. So the question "is there a usable ONNX
Runtime here?" is asked of a **child** -- ``python -c`` running
``embroider.runtime_info()`` -- once per session, and its answer is
cached. Every ONNX node's status line and tooltip reads this module, so
the user gets one sentence in one place instead of one per node:

* "No ONNX Runtime: install Silk's knowledge extra (``okfgraph[cpu]``,
  or ``okfgraph[gpu]`` for CUDA)."
* "CUDA was picked, but this Python has no usable CUDA runtime: install
  ``okfgraph[gpu]``, or pick CPU."
* "ONNX Runtime 1.31.1, CUDA ready" as the quiet line when it is fine.

Nothing here loads a model or starts a host; the answer is what the
*next* ``python -m embroider.serve`` will find in its own interpreter.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, Optional

#: A cold import of embroider plus the DLL probes takes a second or two;
#: a hung child must not hang an evaluation.
PROBE_TIMEOUT = 120.0

#: Printed by the child: one JSON object, never a traceback.
_PROBE = """\
import json
out = {}
try:
    import embroider
    out["embroider"] = getattr(embroider, "__version__", "")
    out["info"] = embroider.runtime_info()
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
try:
    import onnxruntime
    out["onnxruntime"] = onnxruntime.__version__
except Exception:
    pass
print(json.dumps(out, default=str))
"""

_INSTALL = ("install Silk's knowledge extra (okfgraph[cpu], or "
            "okfgraph[gpu] for CUDA)")

_lock = threading.Lock()
_info: Optional[Dict[str, Any]] = None


def _ask(executable: Optional[str]) -> Dict[str, Any]:
    """Run the probe in a child and return its parsed answer."""
    try:
        proc = subprocess.run(
            [executable or sys.executable, "-c", _PROBE],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=PROBE_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return {"error": "the runtime probe timed out"}
    except OSError as exc:
        return {"error": f"the runtime probe could not start: {exc}"}
    text = (proc.stdout or "").strip()
    if proc.returncode != 0 and not text:
        detail = (proc.stderr or "").strip().splitlines()
        return {"error": detail[-1] if detail else
                f"the runtime probe exited with {proc.returncode}"}
    try:
        answer = json.loads(text.splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "the runtime probe printed no answer"}
    return answer if isinstance(answer, dict) else {"error": "bad answer"}


def info(*, refresh: bool = False, executable: Optional[str] = None,
         runner: Optional[Callable[[Optional[str]], Dict[str, Any]]] = None,
         ) -> Dict[str, Any]:
    """The child's answer, computed once and cached.

    ``executable`` picks the interpreter asked (the one ``okf serve`` or
    ``embroider.serve`` will run in); the default is this process's, which
    is what Silk's own hosts use. ``runner`` is the seam the tests use.
    """
    global _info
    with _lock:
        if _info is None or refresh:
            _info = (runner or _ask)(executable)
        return dict(_info)


def forget() -> None:
    """Drop the cached answer, so the next question asks again."""
    global _info
    with _lock:
        _info = None


def cached_text() -> str:
    """``describe()`` once someone has asked, else "" -- for tooltips.

    Building a node must not spawn a child; a tooltip can only improve
    after an evaluation (or a status line) has asked.
    """
    if _info is None:
        return ""
    return describe()


def runtime(answer: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The runtime_info() dict alone (empty when the ask failed)."""
    answer = answer if answer is not None else info()
    found = answer.get("info")
    return found if isinstance(found, dict) else {}


def problem(answer: Optional[Dict[str, Any]] = None,
            device: str = "auto",
            runner: Optional[Callable[[Optional[str]], Dict[str, Any]]] = None,
            ) -> str:
    """One sentence on what will stop a model from loading, or "".

    ``device`` is the node's Device row: a CUDA pick is a promise the
    child has to keep, so a CPU-only runtime is a problem only then.
    """
    answer = answer if answer is not None else info(runner=runner)
    error = str(answer.get("error") or "")
    if error:
        if "No module named" in error and "embroider" in error:
            return f"embroider is not installed: {_INSTALL}."
        return f"The ONNX runtime could not be asked ({error})."
    found = runtime(answer)
    if not found:
        return "The ONNX runtime probe returned no answer."
    if not found.get("ort_dylib_exists"):
        where = found.get("ort_dylib") or "onnxruntime"
        return (f"No ONNX Runtime at {where}: {_INSTALL}.")
    if device == "cuda" and not found.get("cuda_usable"):
        return ("CUDA was picked, but this Python has no usable CUDA ONNX "
                f"Runtime: {_INSTALL}, or pick CPU.")
    return ""


def describe(answer: Optional[Dict[str, Any]] = None,
             runner: Optional[Callable[[Optional[str]], Dict[str, Any]]] = None,
             ) -> str:
    """The quiet status line: what will run the model, or why not."""
    answer = answer if answer is not None else info(runner=runner)
    error = problem(answer)
    if error:
        return error
    found = runtime(answer)
    parts = []
    if answer.get("embroider"):
        parts.append(f"embroider {answer['embroider']}")
    if answer.get("onnxruntime"):
        parts.append(f"ONNX Runtime {answer['onnxruntime']}")
    parts.append("CUDA ready" if found.get("cuda_usable") else "CPU only")
    return ", ".join(parts)


def briefly(answer: Optional[Dict[str, Any]] = None) -> str:
    """``describe()`` in a few words, for a tooltip's last line."""
    answer = answer if answer is not None else info()
    error = problem(answer)
    if error:
        return error
    found = runtime(answer)
    ort = answer.get("onnxruntime") or ""
    head = f"ONNX Runtime {ort}" if ort else "ONNX Runtime"
    return f"{head}, {'CUDA ready' if found.get('cuda_usable') else 'CPU only'}"
