"""CLI rendering for evaluation reward diagnostics."""

from __future__ import annotations

import importlib
from io import StringIO
from types import SimpleNamespace
from typing import TYPE_CHECKING

from rich.console import Console

from hud.cli.utils.display import display_runs
from hud.eval.run import Run

if TYPE_CHECKING:
    import pytest


def _run(reward: float, group_id: str) -> Run:
    run = Run(None, "task", {})
    run.group_id = group_id
    run.grade.reward = reward
    run.trace.status = "completed"
    return run


def test_display_warns_when_variance_exists_only_between_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = [
        _run(0.0, "zero"),
        _run(0.0, "zero"),
        _run(1.0, "one"),
        _run(1.0, "one"),
    ]
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    hud_console_module = importlib.import_module("hud.utils.hud_console")
    monkeypatch.setattr(
        hud_console_module,
        "HUDConsole",
        lambda: SimpleNamespace(console=console),
    )

    display_runs(runs, show_details=False)

    output = stream.getvalue()
    assert "Within-group std: 0.000" in output
    assert "Groups with spread: 0/2" in output
    assert "Constant groups: 2 (1 all-zero, 1 all-one)" in output
    assert "Global reward variance comes entirely from differences between groups" in output
    assert "High variance" not in output
