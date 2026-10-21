"""A JQL subset: comparisons, ``~`` text search, IN, EMPTY, AND/OR/NOT, date functions, ORDER BY."""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from data_sources.base import search_terms
from data_sources.values import parse_date

Value = str | date | float
Reader = Callable[[str], list[Value]]
Predicate = Callable[[Reader], bool]

PRIORITY_RANK = {
    "lowest": 1,
    "trivial": 1,
    "low": 2,
    "minor": 2,
    "medium": 3,
    "major": 4,
    "high": 4,
    "critical": 5,
    "highest": 5,
    "blocker": 6,
}

FIELDS = {
    "key": "key",
    "issuekey": "key",
    "issue": "key",
    "project": "project",
    "summary": "summary",
    "description": "description",
    "comment": "comment",
    "text": "text",
    "status": "status",
    "statuscategory": "statuscategory",
    "type": "type",
    "issuetype": "type",
    "priority": "priority",
    "assignee": "assignee",
    "reporter": "reporter",
    "resolution": "resolution",
    "labels": "labels",
    "label": "labels",
    "sprint": "sprint",
    "parent": "parent",
    "epic link": "parent",
    "created": "created",
    "createddate": "created",
    "updated": "updated",
    "updateddate": "updated",
    "resolved": "resolved",
    "resolutiondate": "resolved",
    "due": "due",
    "duedate": "due",
    "story points": "storypoints",
    "storypoints": "storypoints",
}

_TOKEN = re.compile(
    r"""\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|(!=|!~|>=|<=|=|~|>|<)|([(),])|([^\s(),=!~<>"']+))"""
)
_RELATIVE = re.compile(r"^([+-]?)(\d+)([dwMyhm])$")


class JqlError(ValueError):
    """The query is not JQL this environment understands."""


@dataclass(frozen=True, slots=True)
class Token:
    kind: Literal["string", "op", "punct", "word"]
    text: str


@dataclass(frozen=True, slots=True)
class Query:
    where: Predicate
    order: list[tuple[str, bool]]


def parse(text: str, today: date) -> Query:
    """Parse ``text``; dates such as ``-7d`` and ``startOfMonth()`` are relative to ``today``."""
    return _Parser(_tokenize(text), today).query()


def canonical_field(name: str) -> str:
    key = " ".join(name.lower().replace("_", " ").split())
    if key in FIELDS:
        return FIELDS[key]
    raise JqlError(f"Unknown field {name!r}. Fields: {', '.join(sorted(set(FIELDS.values())))}")


def _tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    position, text = 0, text.strip()
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None or match.end() == position:
            raise JqlError(f"Cannot read the query near {text[position : position + 20]!r}")
        double, single, operator, punct, word = match.groups()
        if double is not None or single is not None:
            tokens.append(
                Token(
                    "string", (double if double is not None else single or "").replace('\\"', '"')
                )
            )
        elif operator:
            tokens.append(Token("op", operator))
        elif punct:
            tokens.append(Token("punct", punct))
        else:
            tokens.append(Token("word", word))
        position = match.end()
        while position < len(text) and text[position].isspace():
            position += 1
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token], today: date) -> None:
        self.tokens = tokens
        self.position = 0
        self.today = today

    def peek(self) -> Token | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> Token:
        token = self.peek()
        if token is None:
            raise JqlError("The query ends too early")
        self.position += 1
        return token

    def at_keyword(self, *words: str) -> bool:
        token = self.peek()
        return token is not None and token.kind == "word" and token.text.upper() in words

    def at_punct(self, punct: str) -> bool:
        token = self.peek()
        return token is not None and token.kind == "punct" and token.text == punct

    def expect_keyword(self, word: str) -> None:
        if not self.at_keyword(word):
            raise JqlError(f"Expected {word}")
        self.take()

    def expect_punct(self, punct: str) -> None:
        if not self.at_punct(punct):
            raise JqlError(f"Expected {punct!r}")
        self.take()

    def query(self) -> Query:
        where: Predicate = _always
        if self.peek() is not None and not self.at_keyword("ORDER"):
            where = self.any_of()
        order: list[tuple[str, bool]] = []
        if self.at_keyword("ORDER"):
            self.take()
            self.expect_keyword("BY")
            while True:
                field = self.field()
                descending = False
                if self.at_keyword("ASC", "DESC"):
                    descending = self.take().text.upper() == "DESC"
                order.append((field, descending))
                if not self.at_punct(","):
                    break
                self.take()
        if (leftover := self.peek()) is not None:
            raise JqlError(f"Unexpected {leftover.text!r}")
        return Query(where=where, order=order)

    def any_of(self) -> Predicate:
        parts = [self.all_of()]
        while self.at_keyword("OR"):
            self.take()
            parts.append(self.all_of())
        return parts[0] if len(parts) == 1 else lambda read: any(part(read) for part in parts)

    def all_of(self) -> Predicate:
        parts = [self.unary()]
        while self.at_keyword("AND"):
            self.take()
            parts.append(self.unary())
        return parts[0] if len(parts) == 1 else lambda read: all(part(read) for part in parts)

    def unary(self) -> Predicate:
        if self.at_keyword("NOT"):
            self.take()
            inner = self.unary()
            return lambda read: not inner(read)
        if self.at_punct("("):
            self.take()
            inner = self.any_of()
            self.expect_punct(")")
            return inner
        return self.clause()

    def field(self) -> str:
        token = self.take()
        if token.kind not in {"word", "string"}:
            raise JqlError(f"Expected a field name, found {token.text!r}")
        return canonical_field(token.text)

    def clause(self) -> Predicate:
        field = self.field()
        token = self.peek()
        if token is None:
            raise JqlError(f"Expected an operator after {field}")
        if token.kind == "op":
            operator = self.take().text
            return _compare(field, operator, self.value(), self.today)
        if self.at_keyword("IN"):
            self.take()
            return _membership(field, self.values(), self.today, negate=False)
        if self.at_keyword("NOT"):
            self.take()
            self.expect_keyword("IN")
            return _membership(field, self.values(), self.today, negate=True)
        if self.at_keyword("IS"):
            self.take()
            negate = self.at_keyword("NOT")
            if negate:
                self.take()
            if not self.at_keyword("EMPTY", "NULL"):
                raise JqlError("Expected EMPTY after IS")
            self.take()
            if negate:
                return lambda read: bool(read(field))
            return lambda read: not read(field)
        raise JqlError(f"Unknown operator {token.text!r}")

    def value(self) -> str | None:
        """A literal; ``None`` stands for EMPTY. Date functions evaluate to ISO dates."""
        token = self.take()
        if token.kind == "string":
            return token.text
        if token.kind != "word":
            raise JqlError(f"Expected a value, found {token.text!r}")
        if self.at_punct("("):
            self.take()
            arguments: list[str] = []
            while not self.at_punct(")"):
                argument = self.take()
                if argument.kind != "punct":
                    arguments.append(argument.text)
            self.take()
            return _function(token.text, arguments, self.today).isoformat()
        if token.text.upper() in {"EMPTY", "NULL"}:
            return None
        return token.text

    def values(self) -> list[str | None]:
        self.expect_punct("(")
        found = [self.value()]
        while self.at_punct(","):
            self.take()
            found.append(self.value())
        self.expect_punct(")")
        return found


def _always(read: Reader) -> bool:
    return True


def _compare(field: str, operator: str, literal: str | None, today: date) -> Predicate:
    if literal is None:
        if operator == "=":
            return lambda read: not read(field)
        if operator == "!=":
            return lambda read: bool(read(field))
        raise JqlError("EMPTY works only with =, !=, IS and IS NOT")
    if operator in {"~", "!~"}:
        terms = search_terms(literal)

        def contains(read: Reader) -> bool:
            return any(all(term in str(value).lower() for term in terms) for value in read(field))

        return contains if operator == "~" else lambda read: not contains(read)
    if operator == "=":
        return lambda read: any(_equal(value, literal, today) for value in read(field))
    if operator == "!=":
        return lambda read: (
            bool(read(field)) and not any(_equal(value, literal, today) for value in read(field))
        )
    return lambda read: any(
        _ordered(field, value, operator, literal, today) for value in read(field)
    )


def _membership(field: str, literals: list[str | None], today: date, *, negate: bool) -> Predicate:
    wants_empty = None in literals
    concrete = [literal for literal in literals if literal is not None]

    def member(read: Reader) -> bool:
        values = read(field)
        if not values:
            return wants_empty
        return any(_equal(value, literal, today) for value in values for literal in concrete)

    if negate:
        return lambda read: bool(read(field)) and not member(read)
    return member


def _equal(value: Value, literal: str, today: date) -> bool:
    if isinstance(value, date):
        return value == _date_literal(literal, today)
    if isinstance(value, int | float):
        try:
            return value == float(literal)
        except ValueError:
            return False
    return value.casefold() == literal.casefold()


def _ordered(field: str, value: Value, operator: str, literal: str, today: date) -> bool:
    left: float
    right: float
    if isinstance(value, date):
        other = _date_literal(literal, today)
        if other is None:
            raise JqlError(f"{literal!r} is not a date; use 2026-03-31, -7d or startOfMonth()")
        left, right = value.toordinal(), other.toordinal()
    elif isinstance(value, int | float):
        try:
            left, right = value, float(literal)
        except ValueError as error:
            raise JqlError(f"{literal!r} is not a number") from error
    elif field == "priority":
        left = PRIORITY_RANK.get(value.lower(), 0)
        right = PRIORITY_RANK.get(literal.lower(), 0)
    elif field == "key":
        project, _, number = value.partition("-")
        other_project, _, other_number = literal.upper().partition("-")
        if project != other_project or not number.isdigit() or not other_number.isdigit():
            return False
        left, right = int(number), int(other_number)
    else:
        raise JqlError(f"{field} cannot be compared with {operator}")
    return {
        ">": left > right,
        ">=": left >= right,
        "<": left < right,
        "<=": left <= right,
    }[operator]


def _date_literal(literal: str, today: date) -> date | None:
    if relative := _RELATIVE.match(literal):
        sign, amount, unit = relative.groups()
        count = -int(amount) if sign == "-" else int(amount)
        return _shift(today, count, unit)
    return parse_date(literal)


def _shift(day: date, count: int, unit: str) -> date:
    if unit == "d":
        return day + timedelta(days=count)
    if unit == "w":
        return day + timedelta(weeks=count)
    if unit == "M":
        month = day.month - 1 + count
        year = day.year + month // 12
        month = month % 12 + 1
        return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))
    if unit == "y":
        return date(
            day.year + count,
            day.month,
            min(day.day, calendar.monthrange(day.year + count, day.month)[1]),
        )
    return day


def _function(name: str, arguments: list[str], today: date) -> date:
    key = name.lower()
    offset = 0
    if arguments:
        relative = _RELATIVE.match(arguments[0])
        try:
            offset = int(relative.group(1) + relative.group(2)) if relative else int(arguments[0])
        except ValueError as error:
            raise JqlError(f"{name}() takes a whole-number offset such as -1") from error
    if key in {"now", "currentdate"}:
        return today
    if key in {"startofday", "endofday"}:
        return today + timedelta(days=offset)
    if key in {"startofweek", "endofweek"}:
        monday = today - timedelta(days=today.weekday()) + timedelta(weeks=offset)
        return monday if key == "startofweek" else monday + timedelta(days=6)
    if key in {"startofmonth", "endofmonth"}:
        first = _shift(today.replace(day=1), offset, "M")
        if key == "startofmonth":
            return first
        return first.replace(day=calendar.monthrange(first.year, first.month)[1])
    if key in {"startofyear", "endofyear"}:
        year = today.year + offset
        return date(year, 1, 1) if key == "startofyear" else date(year, 12, 31)
    if key == "currentuser":
        raise JqlError("currentUser() is not available here; name the person instead")
    raise JqlError(f"Unknown function {name}()")
