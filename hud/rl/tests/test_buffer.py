import random

import pytest

from hud.rl.buffer import ReplayBuffer, SimpleBuffer, create_buffer
from hud.types import Task, Trace


def make_task(task_id: str) -> Task:
    return Task(id=task_id, prompt=f"Prompt {task_id}", mcp_config={"name": "mock"})


def make_trace(task: Task, reward: float, *, is_error: bool = False) -> Trace:
    return Trace(task=task, reward=reward, isError=is_error)


def test_simple_buffer_task_flow() -> None:
    tasks = [make_task("task-0"), make_task("task-1")]
    buffer = SimpleBuffer(tasks, group_size=2)

    batch = buffer.sample_tasks(2)
    assert len(batch) == 4
    assert batch.count(tasks[0]) == 2
    assert batch.count(tasks[1]) == 2

    buffer.add_traces([make_trace(tasks[0], reward) for reward in (1.0, 2.0)])

    retry_batch = buffer.sample_tasks(2)
    assert retry_batch == [tasks[1], tasks[1]]

    buffer.add_traces([make_trace(tasks[1], reward) for reward in (3.0, 4.0)])

    final_batch = buffer.sample_tasks(2)
    assert final_batch == []

    sampled_traces = buffer.sample_traces(2)
    assert len(sampled_traces) == 4
    assert [trace.task.id for trace in sampled_traces] == ["task-0", "task-0", "task-1", "task-1"]

    buffer.reset()
    assert buffer.task_traces["task-0"] == []
    assert buffer.task_traces["task-1"] == []


def test_replay_buffer_trims_history_and_random_sampling() -> None:
    random.seed(42)
    tasks = [make_task("task-0"), make_task("task-1")]
    buffer = ReplayBuffer(tasks, group_size=2, select_strategy="random", buffer_steps=2)

    buffer.add_traces([make_trace(tasks[0], float(idx)) for idx in range(5)])
    buffer.add_traces([make_trace(tasks[1], float(idx)) for idx in range(2)])

    assert len(buffer.task_traces["task-0"]) == 4  # trimmed to buffer_steps * group_size
    assert len(buffer.task_traces["task-1"]) == 2

    sampled = buffer.sample_traces(2)
    assert len(sampled) == 4
    assert {trace.task.id for trace in sampled} == {"task-0", "task-1"}


def test_replay_buffer_variance_prefers_high_variance_tasks() -> None:
    tasks = [make_task("task-0"), make_task("task-1"), make_task("task-2")]
    buffer = ReplayBuffer(tasks, group_size=2, select_strategy="variance", buffer_steps=4)

    buffer.add_traces([make_trace(tasks[0], reward) for reward in (0.0, 5.0, 0.0, 5.0)])
    buffer.add_traces([make_trace(tasks[1], 1.0) for _ in range(4)])
    buffer.add_traces([make_trace(tasks[2], reward) for reward in (2.0, 2.5, 2.0, 2.5)])

    sampled = buffer.sample_traces(1)
    assert len(sampled) == 2
    assert {trace.task.id for trace in sampled} == {"task-0"}


def test_variance_sampling_injects_earlier_traces() -> None:
    task = make_task("task-0")
    buffer = ReplayBuffer([task], group_size=2, select_strategy="variance", buffer_steps=3)

    buffer.add_traces(
        [make_trace(task, reward) for reward in (10.0, 20.0, 1.0, 2.0, 3.0, 4.0)]
    )

    sampled = buffer.sample_traces(1)
    sampled_rewards = {trace.reward for trace in sampled}
    assert sampled_rewards & {10.0, 20.0}


def test_create_buffer_variants() -> None:
    tasks = [make_task("task-0")]

    recent_buffer = create_buffer(tasks, group_size=2, select_strategy="recent")
    assert isinstance(recent_buffer, SimpleBuffer)

    replay_buffer = create_buffer(tasks, group_size=2, select_strategy="random", buffer_steps=2)
    assert isinstance(replay_buffer, ReplayBuffer)

    with pytest.raises(ValueError):
        create_buffer(tasks, group_size=2, select_strategy="unknown")
