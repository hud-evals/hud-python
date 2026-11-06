"""Batches: a task template bound to a dataset, materialized row by row at run time.

A :class:`Batch` names a template, a :class:`Dataset` of rows, and a
:class:`Binding` that turns one row into that template's arguments. Nothing is
stored per row: the scheduler reads rows in bounded windows, mints a
:class:`~hud.eval.Task` for each, runs it through a pool of warm environment
instances, and records one :class:`RowResult`. ``pip install 'hud[batch]'``
brings the Parquet readers.
"""

from __future__ import annotations

from .binding import Binding
from .dataset import Cursor, Dataset, Row
from .materialize import Materialization, RowStream

__all__ = ["Binding", "Cursor", "Dataset", "Materialization", "Row", "RowStream"]
