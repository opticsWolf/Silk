# -*- coding: utf-8 -*-
"""Silk agent suite (opt-in): a local-first agentic runtime as Weave nodes.
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: AGPL-3.0-only

``llama_cpp.server``, plus llama-server's ``POST /tokenize``.

The GGUF pool runs this script instead of ``python -m llama_cpp.server``.
The Python server tokenizes only under ``/extras/tokenize`` with its own
request shape; okf's external lane counts tokens on llama.cpp's
``llama-server`` wire -- ``POST /tokenize`` at the root, ``{content}`` in,
``{tokens}`` out -- and without it can only estimate. This adds that
route and runs the server's own ``main`` unchanged.

Run as a plain script: it imports nothing from Silk, and its own folder
leaves ``sys.path`` first, so Silk's modules there cannot shadow the
server's imports.
"""
import os
import sys

if sys.path and os.path.abspath(sys.path[0] or ".") == os.path.dirname(
        os.path.abspath(__file__)):
    sys.path.pop(0)

from fastapi import Depends  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from llama_cpp.server import app as server_app  # noqa: E402
from llama_cpp.server.__main__ import main  # noqa: E402


class TokenizeRequest(BaseModel):
    """llama-server's ``/tokenize`` body: the text, and whether the
    model's special tokens (BOS) are added -- off, as llama-server does."""

    content: str = ""
    add_special: bool = False


@server_app.router.post("/tokenize", dependencies=[Depends(server_app.authenticate)])
async def tokenize(body: TokenizeRequest,
                   llama_proxy=Depends(server_app.get_llama_proxy)) -> dict:
    tokens = llama_proxy(None).tokenize(
        body.content.encode("utf-8"), add_bos=body.add_special, special=True)
    return {"tokens": tokens}


if __name__ == "__main__":
    main()
