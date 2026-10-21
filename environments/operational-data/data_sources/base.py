"""What every data source shares: tool registration, the task context, and the action log."""

from __future__ import annotations

import json
import re
import shlex
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Annotated, Any, ClassVar, Self

from fastmcp.exceptions import ToolError
from fastmcp.tools import Tool
from pydantic import BaseModel, Field

_TOOL_MARKER = "__data_source_tool__"

Limit = Annotated[int, Field(ge=1, le=500, description="Maximum number of results to return.")]


def tool[F: Callable[..., Any]](fn: F) -> F:
    """Expose a data source method to the agent as an MCP tool named ``<source>_<method>``."""
    setattr(fn, _TOOL_MARKER, True)
    return fn


class ExportError(ValueError):
    """An uploaded file is not an export the data source can read."""


@dataclass(frozen=True, slots=True)
class Action:
    """One change the agent made through a write tool."""

    tool: str
    arguments: dict[str, Any]
    outcome: str


@dataclass(slots=True)
class ActionLog:
    """The changes made through write tools during one task, in order."""

    actions: list[Action] = field(default_factory=list)

    def record(self, tool: str, arguments: Mapping[str, Any], outcome: str) -> None:
        given = {name: value for name, value in arguments.items() if value is not None}
        plain = json.loads(json.dumps(given, default=str))
        self.actions.append(Action(tool=tool, arguments=plain, outcome=outcome))

    def render(self) -> str:
        if not self.actions:
            return "(none)"
        return "\n".join(
            f"{index}. {action.tool}({json.dumps(action.arguments, ensure_ascii=False)})"
            f" -> {action.outcome}"
            for index, action in enumerate(self.actions, start=1)
        )

    def to_json(self) -> list[dict[str, Any]]:
        return [
            {"tool": action.tool, "arguments": action.arguments, "outcome": action.outcome}
            for action in self.actions
        ]


@dataclass(frozen=True, slots=True)
class TaskContext:
    """What a task tells its data sources: the scenario's date and where changes are logged."""

    today: date
    log: ActionLog


class Page[T](BaseModel):
    """The first ``limit`` results of a longer list; ``total`` counts them all."""

    total: int
    items: list[T]


def paginate[T](items: Sequence[T], limit: int) -> Page[T]:
    return Page(total=len(items), items=list(items[:limit]))


class DataSource(ABC):
    """An operational system rebuilt from uploaded exports and served to the agent as tools.

    Subclasses parse their exports in :meth:`load` and mark the methods the agent
    may call with :func:`tool`. Methods that change state call :meth:`record` so
    the grader sees what the agent did. ``summary`` says what to upload and what
    the agent can do with it; the console shows it on the source's upload slot.
    """

    id: ClassVar[str]
    title: ClassVar[str]
    summary: ClassVar[str]

    def __init__(self, context: TaskContext) -> None:
        self.context = context

    @classmethod
    @abstractmethod
    def load(cls, files: Sequence[Path], context: TaskContext) -> Self:
        """Read the uploaded exports; raises :class:`ExportError` for a file that is not one."""

    @abstractmethod
    def describe(self) -> str:
        """What was loaded, in a few words (``4 channels, 63 messages``)."""

    @classmethod
    def tool_prefix(cls) -> str:
        return cls.id.replace("-", "_")

    def tools(self) -> list[Tool]:
        prefix = self.tool_prefix()
        return [
            Tool.from_function(getattr(self, name), name=f"{prefix}_{name}")
            for name, member in vars(type(self)).items()
            if getattr(member, _TOOL_MARKER, False)
        ]

    def record(self, method: str, arguments: Mapping[str, Any], outcome: str) -> None:
        """Log a change made by the write tool ``method``."""
        self.context.log.record(f"{self.tool_prefix()}_{method}", arguments, outcome)


def counted(count: int, singular: str, plural: str | None = None) -> str:
    """``1 channel``, ``3 channels``."""
    return f"{count} {singular if count == 1 else plural or f'{singular}s'}"


def search_terms(query: str | None) -> list[str]:
    """Lowercase search terms; a quoted phrase stays one term."""
    if not query or not query.strip():
        return []
    try:
        parts = shlex.split(query)
    except ValueError:
        parts = query.split()
    return [part.lower() for part in parts if part.strip()]


def contains_all(terms: Sequence[str], *texts: str | None) -> bool:
    haystack = " ".join(text for text in texts if text).lower()
    return all(term in haystack for term in terms)


def normalize_key(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z@.]+", " ", text.lower()).split())


def resolve[T](name: str, options: Mapping[str, T], what: str) -> T:
    """The option ``name`` refers to: an exact key, else the one option whose key contains it.

    Keys are compared normalized, so ``#Finance-Close`` finds ``finance-close``.
    Several keys may name the same option (a handle and an email, say).
    """
    wanted = normalize_key(name.lstrip("#@"))
    keys = {normalize_key(key): key for key in options}
    if wanted in keys:
        return options[keys[wanted]]
    partial = [key for normalized, key in keys.items() if wanted and wanted in normalized]
    matches = list({id(options[key]): options[key] for key in partial}.values())
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ToolError(f"{what} {name!r} is ambiguous: {', '.join(sorted(partial)[:20])}")
    raise ToolError(f"Unknown {what} {name!r}. Known: {', '.join(sorted(options)[:40])}")
