# -*- coding: utf-8 -*-
"""Where a secret lives, which is nowhere near the graph (spec D22).

A node, a preset or a saved graph stores a credential *name*. The value
behind it is resolved when a connection is made, from the environment or
from a file under the user's home directory, and is never written back.
That is what makes a saved graph shareable by construction: there is
nothing in it to leak.

This started inside ``mcp_session``, because MCP servers were the first
thing that needed a key. They are not the last -- D45's remote backends
(litellm, vLLM, a hosted endpoint) need exactly the same rule, and a
second copy of it would be a second place to get it wrong. So the rule
lives here and both import it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from weave.logger import get_logger

log = get_logger("SilkCredentials")

#: Where a credential value may live when it is not in the environment.
#: Outside the graph on purpose (D22).
SECRETS_FILE = Path.home() / ".weave" / "silk" / "secrets.json"


def resolve_credential(name: str) -> Optional[str]:
    """The value behind a credential *name*, or ``None`` (D22).

    Environment first, then the secrets file. Nothing here writes, and the
    value is never handed back to the graph -- only used to build the
    headers of a connection.
    """
    if not name:
        return None
    value = os.environ.get(name)
    if value:
        return value
    try:
        if SECRETS_FILE.is_file():
            data = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                found = data.get(name)
                if isinstance(found, str) and found:
                    return found
    except (OSError, ValueError) as exc:
        log.warning(f"Could not read {SECRETS_FILE}: {exc}")
    return None


def missing_credential(name: str) -> str:
    """The message for a name nobody has set -- it says where to put it."""
    return (
        f"Credential '{name}' is not set. Put it in the environment, or in "
        f"{SECRETS_FILE} -- never in the graph."
    )
