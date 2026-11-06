"""Binding: the rule that turns one dataset row into one task row.

A :class:`Binding` is declarative data, so a batch can be hashed, stored, and
re-run: template args are literals or ``$.column`` paths into the row,
eligibility is a small comparison language compiled to a pyarrow filter (so it
is applied before rows are decoded), and the row id names the column that keys
results. Every minted :class:`~hud.eval.Task` gets ``slug = "<batch_id>:<row_id>"``,
the idempotency key for retries and resume.
"""

from __future__ import annotations

import inspect
import re
from typing import TYPE_CHECKING, Any

import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, Field

from hud.eval.runtime import RuntimeConfig
from hud.eval.task import Task

if TYPE_CHECKING:
    from hud.environment.env import _TaskFactory

    from .dataset import Row

PATH_PREFIX = "$."

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<number>-?\d+(?:\.\d+)?)
      | (?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
      | (?P<word>[A-Za-z_][A-Za-z0-9_.]*)
      | (?P<op><=|>=|==|!=|<|>)
    )""",
    re.VERBOSE,
)
_COMPARISONS = {
    "==": lambda f, v: f == v,
    "!=": lambda f, v: f != v,
    "<": lambda f, v: f < v,
    "<=": lambda f, v: f <= v,
    ">": lambda f, v: f > v,
    ">=": lambda f, v: f >= v,
}
_LITERAL_WORDS: dict[str, Any] = {"true": True, "false": False, "null": None}


def _tokens(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            if text[position:].strip() == "":
                break
            raise ValueError(f"where: cannot parse {text[position:]!r}")
        position = match.end()
        kind = match.lastgroup
        assert kind is not None
        out.append((kind, match.group(kind)))
    return out


def compile_where(text: str) -> Any:
    """Compile ``"<column> <op> <literal> [and|or ...]"`` to a pyarrow expression.

    Comparisons are ``== != < <= > >=`` against a number, a quoted string,
    ``true``/``false``/``null``; ``and`` binds tighter than ``or``. Anything
    richer belongs in a precomputed column, which keeps the filter pushable
    down to the Parquet reader.
    """
    tokens = _tokens(text)
    if not tokens:
        raise ValueError("where: empty expression")
    position = 0

    def next_token() -> tuple[str, str]:
        nonlocal position
        if position >= len(tokens):
            raise ValueError(f"where: unexpected end of expression in {text!r}")
        token = tokens[position]
        position += 1
        return token

    def comparison() -> Any:
        kind, column = next_token()
        if kind != "word" or column in _LITERAL_WORDS or column in {"and", "or"}:
            raise ValueError(f"where: expected a column name, got {column!r} in {text!r}")
        kind, op = next_token()
        if kind != "op":
            raise ValueError(f"where: expected a comparison after {column!r}, got {op!r}")
        kind, raw = next_token()
        if kind == "number":
            literal: Any = float(raw) if "." in raw else int(raw)
        elif kind == "string":
            literal = raw[1:-1].encode().decode("unicode_escape")
        elif kind == "word" and raw in _LITERAL_WORDS:
            literal = _LITERAL_WORDS[raw]
        else:
            raise ValueError(f"where: expected a literal after {op!r}, got {raw!r}")
        if literal is None:
            if op not in ("==", "!="):
                raise ValueError("where: null supports only == and !=")
            return pc.field(column).is_null() if op == "==" else pc.field(column).is_valid()
        return _COMPARISONS[op](pc.field(column), literal)

    def peek_word(word: str) -> bool:
        return position < len(tokens) and tokens[position] == ("word", word)

    def conjunction() -> Any:
        expression = comparison()
        while peek_word("and"):
            next_token()
            expression = expression & comparison()
        return expression

    def disjunction() -> Any:
        expression = conjunction()
        while peek_word("or"):
            next_token()
            expression = expression | conjunction()
        return expression

    expression = disjunction()
    if position != len(tokens):
        raise ValueError(f"where: unexpected {tokens[position][1]!r} in {text!r}")
    return expression


def _where_columns(text: str) -> set[str]:
    words = [value for kind, value in _tokens(text) if kind == "word"]
    return {w for w in words if w not in _LITERAL_WORDS and w not in {"and", "or"}}


def resolve_path(row: Row, path: str) -> Any:
    """``$.a.b`` -> ``row["a"]["b"]``; a missing segment is an error, never ``None``."""
    value: Any = row
    for segment in path[len(PATH_PREFIX) :].split("."):
        if not isinstance(value, dict) or segment not in value:
            raise KeyError(f"binding path {path!r}: no {segment!r} in row")
        value = value[segment]
    return value


def _resolve(spec: Any, row: Row) -> Any:
    if isinstance(spec, str) and spec.startswith(PATH_PREFIX):
        return resolve_path(row, spec)
    if isinstance(spec, dict):
        return {key: _resolve(value, row) for key, value in spec.items()}
    if isinstance(spec, list):
        return [_resolve(value, row) for value in spec]
    return spec


def _paths(spec: Any) -> set[str]:
    if isinstance(spec, str) and spec.startswith(PATH_PREFIX):
        return {spec}
    if isinstance(spec, dict):
        return {path for value in spec.values() for path in _paths(value)}
    if isinstance(spec, list):
        return {path for value in spec for path in _paths(value)}
    return set()


class Binding(BaseModel):
    """How a dataset row becomes a task row.

    ``row_id`` is the column (or ``$.path``) holding the row's stable id.
    ``args`` maps template parameter names to literals or ``$.column`` paths
    (nested dicts and lists resolve recursively). ``where`` keeps only rows the
    expression admits. ``columns`` fills :attr:`Task.columns` the same way;
    ``agent_config`` / ``runtime_config`` ride along onto every task.
    """

    model_config = ConfigDict(frozen=True)

    row_id: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    where: str | None = None
    columns: dict[str, Any] | None = None
    agent_config: dict[str, Any] | None = None
    runtime_config: RuntimeConfig | None = None

    def expression(self) -> Any | None:
        """The ``where`` clause as a pyarrow expression (``None`` when unset)."""
        return compile_where(self.where) if self.where is not None else None

    def required_columns(self) -> set[str]:
        """Top-level row columns this binding reads (ids, args, columns, where)."""
        specs: list[Any] = [self.args, self.columns or {}]
        paths = {path for spec in specs for path in _paths(spec)}
        paths.add(self.row_id if self.row_id.startswith(PATH_PREFIX) else PATH_PREFIX + self.row_id)
        roots = {path[len(PATH_PREFIX) :].split(".", 1)[0] for path in paths}
        if self.where is not None:
            roots |= {column.split(".", 1)[0] for column in _where_columns(self.where)}
        return roots

    def row_id_of(self, row: Row) -> str:
        path = self.row_id if self.row_id.startswith(PATH_PREFIX) else PATH_PREFIX + self.row_id
        value = resolve_path(row, path)
        if value is None or value == "":
            raise ValueError(f"binding row_id {self.row_id!r} is empty for row {row!r}")
        return str(value)

    @staticmethod
    def slug(batch_id: str, row_id: str) -> str:
        return f"{batch_id}:{row_id}"

    def bind(self, batch_id: str, env: str, task_id: str, row: Row) -> Task:
        """Mint the task row for one dataset row."""
        return Task(
            env=env,
            id=task_id,
            slug=self.slug(batch_id, self.row_id_of(row)),
            args=_resolve(self.args, row),
            columns=_resolve(self.columns, row) if self.columns is not None else None,
            agent_config=self.agent_config,
            runtime_config=self.runtime_config,
        )

    def validate_template(self, template: _TaskFactory[...]) -> None:
        """Fail fast when ``args`` and the template's parameters disagree."""
        parameters = template.sig.parameters
        accepts_extra = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        names = {
            name
            for name, p in parameters.items()
            if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        }
        required = {
            name
            for name, p in parameters.items()
            if name in names and p.default is inspect.Parameter.empty
        }
        unknown = set(self.args) - names
        if unknown and not accepts_extra:
            raise ValueError(
                f"binding args {sorted(unknown)} are not parameters of template "
                f"{template.id!r} (accepts {sorted(names)})"
            )
        missing = required - set(self.args)
        if missing:
            raise ValueError(
                f"binding is missing required args {sorted(missing)} of template {template.id!r}"
            )
        if self.where is not None:
            compile_where(self.where)


__all__ = ["PATH_PREFIX", "Binding", "compile_where", "resolve_path"]
