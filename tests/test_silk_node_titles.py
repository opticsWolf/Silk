# -*- coding: utf-8 -*-
"""A node's title is what the canvas shows, not an attribute (G9).

Graph authoring wrote `node.title = ...` and read it back, which
round-tripped through an attribute Weave never renders: the header shows
`node.name` when there is one and its own text otherwise, and
`NodeTitleCommand` writes both. So every agent-placed node was titleless
on screen while the tool reported the title it had asked for.

Typing `nodes/` is what surfaced it -- `BaseControlNode` has no `title`.
These tests pin the reading and writing rules against duck-typed
stand-ins, the way the rest of the canvas seam is tested, so no Qt is
needed.
"""

from __future__ import annotations



from silk.functions.graph_author import (  # noqa: E402
    node_title,
    set_node_title,
)


class _Label:
    def __init__(self, text: str = "", tip: str = ""):
        self._text, self._tip = text, tip

    def toPlainText(self) -> str:
        return self._text

    def toolTip(self) -> str:
        return self._tip

    def setPlainText(self, text: str) -> None:
        self._text = text

    def setToolTip(self, text: str) -> None:
        self._tip = text


class _Header:
    def __init__(self, label):
        self._title = label
        self.laid_out = 0
        self.updated = 0

    def _recalculate_layout(self) -> None:
        self.laid_out += 1

    def update(self) -> None:
        self.updated += 1


class _Node:
    """A node with `name`, like the ones the registry hands back."""

    def __init__(self, name: str = "", label_text: str = ""):
        self.name = name
        self.header = _Header(_Label(label_text))


def test_the_name_is_the_title():
    assert node_title(_Node(name="Planner")) == "Planner"


def test_a_callable_name_is_called():
    node = _Node()
    node.name = lambda: "Called"
    assert node_title(node) == "Called"


def test_the_header_answers_when_there_is_no_name():
    node = _Node(label_text="Header text")
    node.name = ""
    assert node_title(node) == "Header text"


def test_the_tooltip_wins_over_elided_header_text():
    """The header elides; the tooltip keeps the whole thing."""
    node = _Node()
    node.name = ""
    node.header._title = _Label("A very long ti…", "A very long title")
    assert node_title(node) == "A very long title"


def test_a_node_with_neither_is_titleless_not_an_error():
    assert node_title(object()) == ""


def test_renaming_writes_the_model_and_the_header():
    node = _Node(name="old", label_text="old")
    assert set_node_title(node, "new") is True
    assert node.name == "new"
    assert node.header._title.toPlainText() == "new"
    assert node.header._title.toolTip() == "new"
    assert node.header.laid_out == 1, "the header re-lays out or the text is stale"
    assert node_title(node) == "new"


def test_set_name_is_preferred_when_the_node_has_one():
    seen = []

    class WithSetter(_Node):
        def set_name(self, value):
            seen.append(value)

    node = WithSetter(name="old", label_text="old")
    set_node_title(node, "new")
    assert seen == ["new"], "the node's own setter is the model, not the attribute"


def test_a_node_that_refuses_a_rename_says_so():
    class ReadOnly:
        @property
        def name(self):
            return "fixed"

    assert set_node_title(ReadOnly(), "new") is False
