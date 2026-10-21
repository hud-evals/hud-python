"""Tables from CSV, TSV and Excel exports, read through each source's column aliases."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

import openpyxl

from data_sources.base import ExportError
from data_sources.values import parse_date, parse_datetime, parse_number, split_list


@dataclass(frozen=True, slots=True)
class Table:
    """One sheet or delimited file: normalized headers and rows of cell text."""

    name: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def read_tables(path: Path) -> list[Table]:
    """Every table in ``path``: one per CSV file, one per non-empty Excel sheet."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return _excel_tables(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        table = _table(path.stem, _csv_rows(path))
        return [table] if table else []
    raise ExportError(f"{path.name}: expected a CSV or Excel export")


def header_key(header: str) -> str:
    """``Amount (USD)`` and ``amount_usd`` both read as ``amount usd``."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", header.lower()).split())


def cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ") if value.time() != time() else value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _csv_rows(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(text[:8192], ",;\t|")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
    return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text), dialect)]


def _excel_tables(path: Path) -> list[Table]:
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as error:
        raise ExportError(f"{path.name}: not a readable Excel workbook ({error})") from error
    tables: list[Table] = []
    for sheet in workbook.worksheets:
        rows = [[cell_text(value) for value in row] for row in sheet.iter_rows(values_only=True)]
        table = _table(sheet.title, rows)
        if table:
            tables.append(table)
    workbook.close()
    return tables


def _table(name: str, rows: list[list[str]]) -> Table | None:
    """The table under the first row with two or more filled cells (report titles sit above)."""
    start = next((i for i, row in enumerate(rows) if sum(1 for cell in row if cell) >= 2), None)
    if start is None:
        return None
    headers = tuple(header_key(cell) for cell in rows[start])
    width = len(headers)
    body = tuple(
        tuple((row + [""] * width)[:width])
        for row in rows[start + 1 :]
        if any(cell for cell in row)
    )
    return Table(name=name, headers=headers, rows=body)


class Columns:
    """A table's columns located by alias; one canonical name may match repeated columns."""

    def __init__(self, table: Table, aliases: Mapping[str, Sequence[str]]) -> None:
        self.table = table
        self._indexes: dict[str, list[int]] = {}
        for name, names in aliases.items():
            for alias in names:
                key = header_key(alias)
                found = [i for i, header in enumerate(table.headers) if header == key]
                if found:
                    self._indexes[name] = found
                    break

    def has(self, *names: str) -> bool:
        return all(name in self._indexes for name in names)

    def text(self, row: Sequence[str], name: str) -> str | None:
        for index in self._indexes.get(name, ()):
            if row[index]:
                return row[index]
        return None

    def texts(self, row: Sequence[str], name: str) -> list[str]:
        return [row[index] for index in self._indexes.get(name, ()) if row[index]]

    def date(self, row: Sequence[str], name: str) -> date | None:
        return parse_date(self.text(row, name))

    def datetime(self, row: Sequence[str], name: str) -> datetime | None:
        return parse_datetime(self.text(row, name))

    def number(self, row: Sequence[str], name: str) -> float | None:
        return parse_number(self.text(row, name))

    def items(self, row: Sequence[str], name: str) -> list[str]:
        return [item for text in self.texts(row, name) for item in split_list(text)]


def tables_with(
    files: Sequence[Path], aliases: Mapping[str, Sequence[str]], required: Sequence[str]
) -> list[Columns]:
    """The tables in ``files`` that have every ``required`` column."""
    found: list[Columns] = []
    for path in files:
        for table in read_tables(path):
            columns = Columns(table, aliases)
            if columns.has(*required):
                found.append(columns)
    return found
