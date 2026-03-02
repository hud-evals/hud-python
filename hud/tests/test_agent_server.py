from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from hud.agent_server import _build_app


class _FakeEnv:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> _FakeEnv:
        self.entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        _ = (exc_type, exc, tb)
        self.exited = True

    async def list_scenarios(self) -> list[Any]:
        arg = SimpleNamespace(
            name="ticket_id",
            type="string",
            required=True,
            description="Ticket identifier",
            default=None,
        )
        scenario = SimpleNamespace(
            name="demo:investigate",
            short_name="investigate",
            description="Investigate an issue",
            required_args=["ticket_id"],
            arguments=[arg],
        )
        return [scenario]


class _FakeChat:
    def __init__(self) -> None:
        self.trace_id = "trace-test-123"
        self.messages: list[str] = []
        self.last_answer = ""

    async def send(self, message: str) -> Any:
        self.messages.append(message)
        self.last_answer = f"echo:{message}"
        return SimpleNamespace(answer=self.last_answer, tool_calls=[])

    async def send_stream(self, message: str) -> AsyncIterator[Any]:
        self.messages.append(message)
        self.last_answer = f"echo:{message}"
        yield SimpleNamespace(type="text_delta", content=self.last_answer)

    async def finish(self, answer: str | None = None) -> Any:
        final = answer if answer is not None else self.last_answer
        return SimpleNamespace(answer=final, reward=0.75, trace_id=self.trace_id)


def _fake_runner_factory(calls: list[dict[str, Any]]) -> Any:
    @asynccontextmanager
    async def _runner(
        *,
        client: Any,
        model: str,
        env: Any,
        scenario: str,
        args: dict[str, Any],
        max_steps: int,
    ) -> AsyncIterator[_FakeChat]:
        _ = (client, env)
        calls.append(
            {
                "model": model,
                "scenario": scenario,
                "args": args,
                "max_steps": max_steps,
            }
        )
        chat = _FakeChat()
        yield chat

    return _runner


def _make_client(monkeypatch: Any) -> tuple[TestClient, list[dict[str, Any]], _FakeEnv]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("hud.scenario_chat.run_scenario_chat_interactive", _fake_runner_factory(calls))
    env = _FakeEnv()
    app = _build_app(
        env=env,
        client=SimpleNamespace(),
        model="gpt-4o",
        api_key=None,
        session_ttl=120,
    )
    return TestClient(app), calls, env


def test_first_turn_requires_scenario_and_scenario_args(monkeypatch: Any) -> None:
    client, calls, _ = _make_client(monkeypatch)

    with client:
        missing_args = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Begin"}],
                "scenario": "investigate",
            },
        )
        assert missing_args.status_code == 400
        assert "scenario_args is required" in missing_args.text

        ok = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Begin"}],
                "scenario": "investigate",
                "scenario_args": {"ticket_id": "T-1"},
            },
        )
        assert ok.status_code == 200
        payload = ok.json()
        assert payload["hud"]["thread_id"] == payload["hud"]["session_id"]
        assert payload["hud"]["conversation_id"] == payload["hud"]["session_id"]
        assert calls[0]["args"] == {"ticket_id": "T-1"}


def test_followup_allows_thread_id_body_alias(monkeypatch: Any) -> None:
    client, _, _ = _make_client(monkeypatch)

    with client:
        first = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Begin"}],
                "scenario": "investigate",
                "scenario_args": {"ticket_id": "T-2"},
            },
        )
        session_id = first.json()["hud"]["session_id"]

        follow_up = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "What happened?"}],
                "thread_id": session_id,
            },
        )
        assert follow_up.status_code == 200
        assert follow_up.json()["choices"][0]["message"]["content"] == "echo:What happened?"


def test_lifecycle_tool_calls_and_mcp_surface(monkeypatch: Any) -> None:
    client, _, _ = _make_client(monkeypatch)

    with client:
        tool_defs = client.get("/v1/lifecycle-tools")
        assert tool_defs.status_code == 200
        names = {tool["name"] for tool in tool_defs.json()["tools"]}
        assert {"scenario_list", "scenario_start", "scenario_send", "scenario_finish"} <= names

        mcp_defs = client.get("/mcp/tools")
        assert mcp_defs.status_code == 200
        mcp_names = {tool["name"] for tool in mcp_defs.json()["tools"]}
        assert names == mcp_names

        start = client.post(
            "/v1/lifecycle-tools/call",
            json={
                "name": "scenario_start",
                "arguments": {
                    "scenario": "investigate",
                    "scenario_args": {"ticket_id": "T-3"},
                    "message": "Begin with context",
                },
            },
        )
        assert start.status_code == 200
        hud = start.json()["hud"]
        session_id = hud["session_id"]

        send = client.post(
            "/mcp/tools/call",
            json={
                "name": "scenario_send",
                "arguments": {"thread_id": session_id, "message": "Follow up"},
            },
        )
        assert send.status_code == 200
        assert send.json()["answer"] == "echo:Follow up"

        finish = client.post(
            "/mcp/tools/call",
            json={
                "name": "scenario_finish",
                "arguments": {"conversation_id": session_id, "answer": "Final answer"},
            },
        )
        assert finish.status_code == 200
        assert finish.json()["answer"] == "Final answer"
        assert finish.json()["reward"] == 0.75


def test_legacy_finish_endpoint_and_session_listing(monkeypatch: Any) -> None:
    client, _, env = _make_client(monkeypatch)

    with client:
        first = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Begin"}],
                "scenario": "investigate",
                "scenario_args": {"ticket_id": "T-4"},
            },
        )
        session_id = first.json()["hud"]["session_id"]

        sessions = client.get("/v1/sessions")
        assert sessions.status_code == 200
        assert sessions.json()["sessions"][0]["thread_id"] == session_id
        assert sessions.json()["sessions"][0]["conversation_id"] == session_id

        done = client.post(f"/v1/sessions/{session_id}/finish")
        assert done.status_code == 200
        assert done.json()["session_id"] == session_id
        assert done.json()["thread_id"] == session_id
        assert done.json()["conversation_id"] == session_id

    assert env.entered is True
    assert env.exited is True
