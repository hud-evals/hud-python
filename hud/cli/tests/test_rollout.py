from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
import typer

import hud.cli.rollout as rollout_cli
from hud.rl.schema import RolloutRecord
from hud.types import AgentType

if TYPE_CHECKING:
    from pathlib import Path


@patch("hud.cli.rollout.write_rollouts_jsonl")
@patch("hud.cli.rollout.collect_rollouts", new_callable=AsyncMock)
@patch("hud.cli.rollout._resolve_agent")
def test_collect_command_happy_path(
    mock_resolve_agent,
    mock_collect_rollouts,
    mock_write_rollouts,
    tmp_path: Path,
) -> None:
    mock_resolve_agent.return_value = (AgentType.CLAUDE, {"model": "x"})
    mock_collect_rollouts.return_value = [
        RolloutRecord(
            rollout_id="rollout_1",
            source="tasks.json",
            task_index=0,
            repeat_index=0,
            prompt="Prompt",
        )
    ]
    mock_write_rollouts.return_value = tmp_path / "rollouts.jsonl"

    rollout_cli.collect_command(
        source="tasks.json",
        output=tmp_path / "rollouts.jsonl",
        agent=AgentType.CLAUDE,
        model=None,
        allowed_tools="act,observe",
        disallowed_tools=None,
        max_concurrent=5,
        max_steps=7,
        group_size=2,
        split="train",
        verbose=False,
    )

    mock_resolve_agent.assert_called_once()
    mock_collect_rollouts.assert_awaited_once()
    assert mock_collect_rollouts.call_args.kwargs["source"] == "tasks.json"
    assert mock_collect_rollouts.call_args.kwargs["group_size"] == 2
    mock_write_rollouts.assert_called_once()


@patch("hud.cli.utils.tasks.find_tasks_file", side_effect=FileNotFoundError())
def test_collect_command_exits_without_source(mock_find_tasks_file) -> None:
    with pytest.raises(typer.Exit):
        rollout_cli.collect_command(source=None)

    mock_find_tasks_file.assert_called_once()
