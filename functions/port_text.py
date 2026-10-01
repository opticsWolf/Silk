# -*- coding: utf-8 -*-
"""Text read from a String input port, without truth-testing it.

A String port also accepts pass-through wires: List, Tuple, Set, Dict,
NdArray, DataFrame and PILImage values cross into it unchanged
(``_PASS_THROUGH`` in Weave's port registry). The usual
``str(inputs.get(key) or default)`` then fails twice over: truth-testing
an array or a DataFrame raises, and ``str()`` of a container is a repr,
which would go out as a command, a URL or a prompt.
"""

from typing import Any, Mapping


def text_input(inputs: Mapping[str, Any], key: str, default: str = "") -> str:
    """Input *key* as stripped text; *default* when absent or blank.

    Raises:
        ValueError: when the value is not text, naming the input and what
            arrived. Callers report it the way they report any unreadable
            input; a node's compute never lets it escape.
    """
    value = inputs.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"'{key}' needs text, got {type(value).__name__}")
    return value.strip() or default
