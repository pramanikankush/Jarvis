"""Declarative tool registry (spec: "make tools easy to add later").

A tool is an async method on the Agent, decorated with @register. The registry
is the single source of truth for tool metadata: the agent's decision prompt
(TOOL_DOCS) and the dispatcher both derive from it. Adding a tool means:

    1. write one async method ``async def _tool_my_tool(self, args)`` on Agent
    2. decorate it with ``@registry.register("my_tool", "description", "category")``

It then appears in the LLM's tool list automatically and is callable by name.
No other file changes needed. No multi-agent machinery — this is just the
"modular tool architecture" seam the spec asks for, kept deliberately thin.
"""
from __future__ import annotations


class Tool:
    __slots__ = ("name", "description", "category")

    def __init__(self, name: str, description: str, category: str = "general"):
        self.name = name
        self.description = description
        self.category = category


TOOLS: dict[str, Tool] = {}


def register(name: str, description: str, category: str = "general"):
    """Decorator for an Agent tool-handler method. Registers its metadata."""

    def deco(method):
        TOOLS[name] = Tool(name, description, category)
        return method

    return deco


def get(name: str) -> Tool | None:
    return TOOLS.get(name)


def all_tools() -> dict[str, Tool]:
    return dict(TOOLS)
