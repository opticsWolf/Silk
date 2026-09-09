# -*- coding: utf-8 -*-
"""The error-family hooks, which could not run at all (D15).

`Hooks.on_model_request_error` and its three siblings are the recover-or-
propagate handlers: a registered callback returns a replacement result to
recover, or raises to let the failure through. All four read the registry
directly and called what they found — but the registry holds `HookEntry`
records (the callback plus what it declares about itself, D13/D14), not
bare callables. Calling the record raised `TypeError` *inside the handler
that exists to recover from an error*, turning every recoverable failure
into an unrelated crash.

Found by mypy once `functions/` was made to typecheck cleanly, which is
the argument for keeping it clean.
"""

from __future__ import annotations

import asyncio

import pytest


from silk.functions.hooks import (
    HOOK_ON_MODEL_REQUEST_ERROR,
    HOOK_ON_OUTPUT_PROCESS_ERROR,
    HOOK_ON_OUTPUT_VALIDATE_ERROR,
    HOOK_ON_TOOL_EXECUTE_ERROR,
    Hooks,
)


def _hooks():
    return Hooks()


def test_a_registered_handler_can_recover_a_failed_model_request():
    hooks = _hooks()
    hooks._registry.register(
        HOOK_ON_MODEL_REQUEST_ERROR, lambda **kw: "recovered")

    result = asyncio.run(
        hooks.on_model_request_error(ctx=None, error=RuntimeError("boom")))

    assert result == "recovered"


def test_a_handler_that_returns_nothing_lets_the_error_through():
    hooks = _hooks()
    seen = []
    hooks._registry.register(
        HOOK_ON_MODEL_REQUEST_ERROR,
        lambda **kw: seen.append(kw.get("error")) and None)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            hooks.on_model_request_error(ctx=None, error=RuntimeError("boom")))

    assert len(seen) == 1, "the handler still ran"


def test_the_handler_is_given_what_the_event_carries():
    hooks = _hooks()
    got = {}

    def handler(**kwargs):
        got.update(kwargs)
        return "ok"

    hooks._registry.register(HOOK_ON_TOOL_EXECUTE_ERROR, handler)
    error = ValueError("no such path")

    result = asyncio.run(hooks.on_tool_execute_error(
        ctx="the-context", tool_name="read_file", error=error))

    assert result == "ok"
    assert got == {"ctx": "the-context", "tool_name": "read_file",
                   "error": error}


@pytest.mark.parametrize("event,call", [
    (HOOK_ON_OUTPUT_VALIDATE_ERROR, "on_output_validate_error"),
    (HOOK_ON_OUTPUT_PROCESS_ERROR, "on_output_process_error"),
])
def test_the_output_handlers_recover_the_same_way(event, call):
    hooks = _hooks()
    hooks._registry.register(event, lambda **kw: {"fixed": True})

    result = asyncio.run(getattr(hooks, call)(
        ctx=None, output="junk", error=ValueError("bad")))

    assert result == {"fixed": True}


def test_an_async_handler_is_awaited():
    hooks = _hooks()

    async def handler(**kwargs):
        return "async recovery"

    hooks._registry.register(HOOK_ON_MODEL_REQUEST_ERROR, handler)

    result = asyncio.run(
        hooks.on_model_request_error(ctx=None, error=RuntimeError("boom")))

    assert result == "async recovery"


def test_with_nothing_registered_the_error_propagates_unchanged():
    hooks = _hooks()
    error = RuntimeError("boom")

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(hooks.on_model_request_error(ctx=None, error=error))

    assert caught.value is error
