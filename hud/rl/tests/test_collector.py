from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from hud.rl.collector import (
    build_rollout_records,
    collect_rollouts,
    write_rollouts_jsonl,
)
from hud.rl.schema import make_rollout_id
from hud.types import Trace, TraceStep


def test_make_rollout_id_is_stable() -> None:
    first = make_rollout_id("source", 1, 2, "prompt")
    second = make_rollout_id("source", 1, 2, "prompt")
    changed = make_rollout_id("source", 1, 3, "prompt")

    assert first == second
    assert first != changed
    assert first.startswith("rollout_")


def test_build_rollout_records_with_group_size_and_errors() -> None:
    task = {"id": "task-1", "prompt": "Solve 2+2", "mcp_config": {"local": {"url": "x"}}}
    trace = Trace(
        reward=1.0,
        done=True,
        content="4",
        trace=[TraceStep(category="agent", request={"prompt": "Solve 2+2"}, result={"text": "4"})],
    )

    records = build_rollout_records(
        source="tasks.json",
        tasks=[task],
        results=[trace, None],
        group_size=2,
    )

    assert len(records) == 2
    assert records[0].task_index == 0
    assert records[0].repeat_index == 0
    assert records[0].prompt == "Solve 2+2"
    assert records[0].trace[0]["category"] == "agent"

    assert records[1].task_index == 0
    assert records[1].repeat_index == 1
    assert records[1].is_error is True
    assert records[1].content == "No trace returned"


@pytest.mark.asyncio
async def test_collect_rollouts_repeats_tasks_and_uses_split() -> None:
    raw_task = {"prompt": "Task prompt", "mcp_config": {"local": {"url": "http://localhost"}}}
    trace = Trace(reward=0.5, done=True, content="done")

    with (
        patch("hud.rl.collector.load_tasks", return_value=[raw_task]) as mock_load_tasks,
        patch("hud.rl.collector.run_dataset", new_callable=AsyncMock) as mock_run_dataset,
    ):
        mock_run_dataset.return_value = [trace, trace]

        records = await collect_rollouts(
            name="test-rollout",
            source="hud-evals/demo",
            agent_type="claude",
            group_size=2,
            split="test",
            max_steps=3,
            max_concurrent=7,
            metadata={"suite": "demo"},
            auto_respond=True,
        )

    mock_load_tasks.assert_called_once_with("hud-evals/demo:test", raw=True)
    called_dataset = mock_run_dataset.call_args.args[0]
    assert len(called_dataset) == 2
    assert called_dataset[0]["prompt"] == "Task prompt"
    assert len(records) == 2
    assert records[0].reward == 0.5


@pytest.mark.asyncio
async def test_collect_rollouts_passes_auto_respond_into_agent_params() -> None:
    raw_task = {"prompt": "Task prompt", "mcp_config": {"local": {"url": "http://localhost"}}}

    with (
        patch("hud.rl.collector.load_tasks", return_value=[raw_task]),
        patch("hud.rl.collector.run_dataset", new_callable=AsyncMock) as mock_run_dataset,
    ):
        mock_run_dataset.return_value = [Trace(reward=1.0, done=True)]

        await collect_rollouts(
            name="test-rollout",
            source="hud-evals/demo",
            agent_type="claude",
            agent_params={"model": "claude-sonnet-4-5"},
            auto_respond=False,
        )

    params = mock_run_dataset.call_args.kwargs["agent_params"]
    assert params["model"] == "claude-sonnet-4-5"
    assert params["auto_respond"] is False


@pytest.mark.asyncio
async def test_collect_rollouts_raises_on_non_dict_entries_in_local_json(tmp_path) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps([{"prompt": "ok", "mcp_config": {"local": {"url": "x"}}}, "not-an-object"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="item 1: expected object"):
        await collect_rollouts(
            name="test-rollout",
            source=str(tasks_path),
            agent_type="claude",
        )


def test_write_rollouts_jsonl(tmp_path) -> None:
    record = build_rollout_records(
        source="tasks.json",
        tasks=[{"prompt": "A", "mcp_config": {"x": 1}}],
        results=[Trace(reward=1.0, done=True)],
        group_size=1,
    )[0]
    output_path = tmp_path / "rollouts" / "data.jsonl"

    written = write_rollouts_jsonl([record], output_path)

    assert written == output_path
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = cast("dict[str, Any]", json.loads(lines[0]))
    assert payload["schema_version"] == "hud.rollout.v1"
    assert payload["task_index"] == 0
