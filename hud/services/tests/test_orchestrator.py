from __future__ import annotations

from typing import Any

import pytest

from hud.eval.task import Task
from hud.native.chat import env as chat_env
from hud.services import ChatDefinition
from hud.services.orchestrator import OrchestratorExecutor, _build_orchestrator_env
from hud.types import Trace


class FakeQueue:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


class FakeContext:
    def __init__(
        self,
        text: str,
        *,
        context_id: str = "ctx-1",
        task_id: str = "task-1",
        message_id: str = "msg-1",
    ) -> None:
        self.context_id = context_id
        self.task_id = task_id
        self.message = type("Msg", (), {"message_id": message_id})
        self._text = text

    def get_user_input(self) -> str:
        return self._text


def _definitions() -> list[ChatDefinition]:
    return [
        ChatDefinition(
            name="chat",
            task=chat_env("chat_full"),
            model="gpt-4o",
            description="General chat.",
        ),
    ]


def test_build_orchestrator_env_registers_scenario_and_tools() -> None:
    env = _build_orchestrator_env(_definitions())
    assert "orchestrate" in env._scenarios


def test_orchestrator_init_builds_env_and_task() -> None:
    orch = OrchestratorExecutor(_definitions(), main_model="gpt-4o")
    assert orch._env is not None
    assert orch._main_task is not None
    assert orch._main_task.scenario == "orchestrate"


def test_orchestrator_requires_definitions() -> None:
    with pytest.raises(ValueError, match="At least one"):
        OrchestratorExecutor([], main_model="gpt-4o")


def test_agent_card_lists_skills() -> None:
    defs = [
        ChatDefinition(name="a", task=chat_env("chat_full"), model="m", description="A"),
        ChatDefinition(name="b", task=chat_env("chat_simple"), model="m", description="B"),
    ]
    orch = OrchestratorExecutor(defs, main_model="gpt-4o")
    card = orch.agent_card()
    skill_ids = [s.id for s in card.skills]
    assert "a" in skill_ids
    assert "b" in skill_ids


@pytest.mark.asyncio
async def test_execute_emits_working_and_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = OrchestratorExecutor(_definitions(), main_model="gpt-4o")
    queue = FakeQueue()
    context = FakeContext("hello")

    async def _fake_send(msg: Any) -> Trace:
        return Trace(content="done")

    chat = orch._get_or_create_chat("ctx-1")
    monkeypatch.setattr(chat, "send", _fake_send)
    # Re-insert so execute finds the monkeypatched chat
    orch._sessions["ctx-1"] = chat

    await orch.execute(context, queue)

    assert len(queue.events) == 2
    assert queue.events[0].status.state.value == "working"
    assert queue.events[1].status.state.value == "completed"


@pytest.mark.asyncio
async def test_execute_maps_errors_to_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = OrchestratorExecutor(_definitions(), main_model="gpt-4o")
    queue = FakeQueue()
    context = FakeContext("hello")

    async def _fail(msg: Any) -> Trace:
        raise RuntimeError("boom")

    chat = orch._get_or_create_chat("ctx-1")
    monkeypatch.setattr(chat, "send", _fail)
    orch._sessions["ctx-1"] = chat

    await orch.execute(context, queue)

    assert len(queue.events) == 2
    assert queue.events[-1].status.state.value == "failed"
    assert "boom" in queue.events[-1].status.message.parts[0].root.text


@pytest.mark.asyncio
async def test_cancel_clears_session() -> None:
    orch = OrchestratorExecutor(_definitions(), main_model="gpt-4o")
    orch._get_or_create_chat("ctx-1")
    assert "ctx-1" in orch._sessions

    queue = FakeQueue()
    context = FakeContext("", context_id="ctx-1", task_id="t")
    await orch.cancel(context, queue)

    assert "ctx-1" not in orch._sessions
    assert queue.events[-1].status.state.value == "canceled"


def test_get_or_create_reuses_session() -> None:
    orch = OrchestratorExecutor(_definitions(), main_model="gpt-4o")
    c1 = orch._get_or_create_chat("ctx-1")
    c2 = orch._get_or_create_chat("ctx-1")
    assert c1 is c2
